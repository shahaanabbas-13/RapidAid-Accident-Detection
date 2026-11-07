"""Interpretable motion measurements derived from tracked vehicle boxes.

These image-plane measurements are evidence for review. They are not calibrated
physical speeds and must not be interpreted as accident probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "frame_index",
    "time_seconds",
    "vehicle_count",
    "median_pixel_speed_per_second",
    "p95_pixel_speed_per_second",
    "median_pixel_acceleration_per_second2",
    "max_pixel_acceleration_per_second2",
]


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
