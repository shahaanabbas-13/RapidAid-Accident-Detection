"""Interpretable motion measurements derived from tracked vehicle boxes.

These image-plane measurements are evidence for review. They are not calibrated
physical speeds and must not be interpreted as accident probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cv2


SIGNAL_COLUMNS = [
    "frame_index",
    "time_seconds",
    "vehicle_count",
    "median_pixel_speed_per_second",
    "p95_pixel_speed_per_second",
    "median_pixel_acceleration_per_second2",
    "max_pixel_acceleration_per_second2",
]
FLOW_COLUMNS = [
    "frame_index",
    "time_seconds",
    "mean_flow_px_per_sample",
    "p95_flow_px_per_sample",
    "high_motion_pixel_fraction",
]
GEOMETRY_TRACK_COLUMNS = [
    "frame_index",
    "time_seconds",
    "overlap_pair_count",
    "max_bbox_iou",
    "one_step_track_loss_count",
    "one_step_track_loss_fraction",
    "likely_edge_exit_count",
]


def extract_geometry_track_loss_signals(
    detections: pd.DataFrame,
    *,
    fps: float,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    sample_every: int = 1,
    overlap_iou_threshold: float = 0.1,
    border_margin_fraction: float = 0.05,
) -> pd.DataFrame:
    """Summarize bbox overlap and one-step losses of tracked IDs per sample.

    A missing track is a continuity change, not proof that a vehicle vanished.
    Border proximity marks possible exits. IoU measures image-box overlap only;
    neither signal establishes a physical collision.
    """
    if fps <= 0 or frame_count < 0 or sample_every < 1 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("fps, sample_every, and frame dimensions must be positive; frame_count cannot be negative")
    if not 0 <= overlap_iou_threshold <= 1:
        raise ValueError("overlap_iou_threshold must be between 0 and 1")
    if not 0 <= border_margin_fraction < 0.5:
        raise ValueError("border_margin_fraction must be between 0 and 0.5")

    frame_indices = np.arange(0, frame_count, sample_every, dtype=int)
    result = pd.DataFrame({"frame_index": frame_indices})
    result["time_seconds"] = result["frame_index"] / fps
    for column in GEOMETRY_TRACK_COLUMNS[2:]:
        result[column] = (
            0.0
            if column in {"max_bbox_iou", "one_step_track_loss_fraction"}
            else 0
        )
    result = result.set_index("frame_index")
    required = {"frame_index", "track_id", "x1", "y1", "x2", "y2"}
    if not len(result) or detections.empty or not required.issubset(detections.columns):
        return result.reset_index()[GEOMETRY_TRACK_COLUMNS]

    rows = detections.loc[detections["track_id"].notna()].copy()
    rows = rows.drop_duplicates(["frame_index", "track_id"], keep="last")
    grouped = {int(index): group for index, group in rows.groupby("frame_index")}
    margin_x = frame_width * border_margin_fraction
    margin_y = frame_height * border_margin_fraction
    previous_ids: set[object] = set()
    previous_boxes: dict[object, tuple[float, float, float, float]] = {}

    for frame_index in frame_indices:
        current = grouped.get(int(frame_index))
        current_ids: set[object] = set()
        current_boxes: dict[object, tuple[float, float, float, float]] = {}
        if current is not None:
            for row in current.itertuples(index=False):
                track_id = row.track_id
                box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
                current_ids.add(track_id)
                current_boxes[track_id] = box

        if len(current_boxes) > 1:
            boxes = np.asarray(list(current_boxes.values()), dtype=float)
            overlaps: list[float] = []
            for first in range(len(boxes) - 1):
                x1 = np.maximum(boxes[first, 0], boxes[first + 1 :, 0])
                y1 = np.maximum(boxes[first, 1], boxes[first + 1 :, 1])
                x2 = np.minimum(boxes[first, 2], boxes[first + 1 :, 2])
                y2 = np.minimum(boxes[first, 3], boxes[first + 1 :, 3])
                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                area_first = max(0, boxes[first, 2] - boxes[first, 0]) * max(0, boxes[first, 3] - boxes[first, 1])
                area_other = np.maximum(0, boxes[first + 1 :, 2] - boxes[first + 1 :, 0]) * np.maximum(
                    0, boxes[first + 1 :, 3] - boxes[first + 1 :, 1]
                )
                union = area_first + area_other - intersection
                ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
                overlaps.extend(ious.tolist())
            if overlaps:
                result.loc[frame_index, "overlap_pair_count"] = int(
                    sum(value >= overlap_iou_threshold for value in overlaps)
                )
                result.loc[frame_index, "max_bbox_iou"] = float(max(overlaps))

        lost_ids = previous_ids - current_ids
        result.loc[frame_index, "one_step_track_loss_count"] = len(lost_ids)
        if previous_ids:
            result.loc[frame_index, "one_step_track_loss_fraction"] = len(lost_ids) / len(previous_ids)
        result.loc[frame_index, "likely_edge_exit_count"] = sum(
            1
            for track_id in lost_ids
            if (
                previous_boxes[track_id][0] <= margin_x
                or previous_boxes[track_id][1] <= margin_y
                or previous_boxes[track_id][2] >= frame_width - margin_x
                or previous_boxes[track_id][3] >= frame_height - margin_y
            )
        )
        previous_ids = current_ids
        previous_boxes = current_boxes

    return result.reset_index()[GEOMETRY_TRACK_COLUMNS]


def extract_optical_flow_signals(
    video_path: str,
    *,
    fps: float,
    frame_count: int,
    sample_every: int = 1,
    max_width: int = 640,
    motion_threshold_px: float = 1.5,
) -> pd.DataFrame:
    """Measure dense Farneback optical flow between sampled video frames.

    Flow is computed on downscaled grayscale frames to limit CPU cost. Values
    describe apparent whole-frame image motion; camera shake and background
    movement contribute, so this is not a vehicle-only or crash score.
    """
    if fps <= 0 or frame_count < 0 or sample_every < 1 or max_width < 32:
        raise ValueError("fps, sample_every, and max_width must be positive; frame_count cannot be negative")
    if motion_threshold_px < 0:
        raise ValueError("motion_threshold_px cannot be negative")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video for optical-flow extraction: {video_path}")

    measurements: list[dict[str, int | float]] = []
    previous_gray: np.ndarray | None = None
    frame_index = 0
    try:
        while frame_index < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                height, width = frame.shape[:2]
                scale = min(1.0, max_width / width)
                resized = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                )
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                if previous_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        previous_gray,
                        gray,
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
                    measurements.append({
                        "frame_index": frame_index,
                        "time_seconds": frame_index / fps,
                        "mean_flow_px_per_sample": float(np.mean(magnitude)),
                        "p95_flow_px_per_sample": float(np.percentile(magnitude, 95)),
                        "high_motion_pixel_fraction": float(np.mean(magnitude >= motion_threshold_px)),
                    })
                previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    return pd.DataFrame(measurements, columns=FLOW_COLUMNS)


def extract_tracking_signals(
    detections: pd.DataFrame,
    *,
    fps: float,
    frame_count: int,
    sample_every: int = 1,
) -> pd.DataFrame:
    """Summarize vehicle count and image-plane motion at each sampled frame.

    Track speeds use the change in bounding-box center between observations.
    Acceleration is the change in that image-plane velocity. Missed/changed
    track IDs and camera perspective can affect these values.
    """
    if fps <= 0 or frame_count < 0 or sample_every < 1:
        raise ValueError("fps and sample_every must be positive; frame_count cannot be negative")

    frame_indices = np.arange(0, frame_count, sample_every, dtype=int)
    if not len(frame_indices):
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    base = pd.DataFrame({"frame_index": frame_indices})
    base["time_seconds"] = base["frame_index"] / fps
    base["vehicle_count"] = 0
    base["median_pixel_speed_per_second"] = np.nan
    base["p95_pixel_speed_per_second"] = np.nan
    base["median_pixel_acceleration_per_second2"] = np.nan
    base["max_pixel_acceleration_per_second2"] = np.nan

    required = {"frame_index", "track_id", "x1", "y1", "x2", "y2"}
    if detections.empty or not required.issubset(detections.columns):
        return base[SIGNAL_COLUMNS]

    rows = detections.loc[detections["track_id"].notna()].copy()
    if rows.empty:
        return base[SIGNAL_COLUMNS]
    rows["center_x"] = (rows["x1"] + rows["x2"]) / 2.0
    rows["center_y"] = (rows["y1"] + rows["y2"]) / 2.0
    rows = rows.sort_values(["track_id", "frame_index"])

    counts = rows.groupby("frame_index")["track_id"].nunique()
    base["vehicle_count"] = base["frame_index"].map(counts).fillna(0).astype(int)

    frame_step_seconds = sample_every / fps
    rows["delta_x"] = rows.groupby("track_id")["center_x"].diff()
    rows["delta_y"] = rows.groupby("track_id")["center_y"].diff()
    rows["pixel_speed_per_second"] = np.hypot(rows["delta_x"], rows["delta_y"]) / frame_step_seconds
    rows["pixel_acceleration_per_second2"] = (
        rows.groupby("track_id")["pixel_speed_per_second"].diff() / frame_step_seconds
    ).abs()

    speed = rows.groupby("frame_index")["pixel_speed_per_second"]
    acceleration = rows.groupby("frame_index")["pixel_acceleration_per_second2"]
    stats = pd.DataFrame({
        "median_pixel_speed_per_second": speed.median(),
        "p95_pixel_speed_per_second": speed.quantile(0.95),
        "median_pixel_acceleration_per_second2": acceleration.median(),
        "max_pixel_acceleration_per_second2": acceleration.max(),
    })
    base = base.set_index("frame_index")
    for column in stats.columns:
        base[column] = stats[column].reindex(base.index)
    return base.reset_index()[SIGNAL_COLUMNS]
