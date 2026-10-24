"""Phase 1 video metadata and vehicle-tracking baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd
import torch
from ultralytics import YOLO


VEHICLE_CLASS_IDS = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def inspect_video(video_path: str | Path) -> VideoInfo:
    """Read basic container metadata and reject files OpenCV cannot decode."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Could not open this video. Try MP4 (H.264) or AVI.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
            raise ValueError("Video metadata is incomplete or the codec is unsupported.")
        return VideoInfo(width, height, fps, frame_count, frame_count / fps)
    finally:
        capture.release()


def analyze_video(
    video_path: str | Path,
    output_video: str | Path,
    *,
    model_path: str = "yolo11n.pt",
    sample_every: int = 2,
    device: str | int = "auto",
    progress: Callable[[float], None] | None = None,
) -> tuple[pd.DataFrame, VideoInfo, str]:
    """Track COCO vehicles and save an annotated review copy plus detections CSV.

    The output is vehicle evidence only. This function does not classify crashes.
    """
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    video_path = Path(video_path)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    info = inspect_video(video_path)
    selected_device: str | int = (0 if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = YOLO(model_path)
    writer: cv2.VideoWriter | None = None
    rows: list[dict[str, int | float | str | None]] = []

    try:
        stream = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASS_IDS),
            conf=0.25,
            imgsz=640,
            device=selected_device,
            vid_stride=sample_every,
            verbose=False,
        )
        for sampled_index, result in enumerate(stream):
            frame = result.plot()
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_video), fourcc, max(info.fps / sample_every, 1.0), (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create annotated MP4 output.")
            writer.write(frame)

            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for box, confidence, class_id, track_id in zip(xyxy, confs, classes, ids):
                    rows.append({
                        "time_seconds": round(sampled_index * sample_every / info.fps, 3),
                        "frame_index": sampled_index * sample_every,
                        "track_id": track_id,
                        "class_id": int(class_id),
                        "class_name": result.names[int(class_id)],
                        "confidence": round(float(confidence), 4),
                        "x1": round(float(box[0]), 2),
                        "y1": round(float(box[1]), 2),
                        "x2": round(float(box[2]), 2),
                        "y2": round(float(box[3]), 2),
                    })
            if progress:
                progress(min((sampled_index + 1) * sample_every / info.frame_count, 0.999))
    finally:
        if writer is not None:
            writer.release()

    if progress:
        progress(1.0)
    columns = ["time_seconds", "frame_index", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=columns), info, str(selected_device)
