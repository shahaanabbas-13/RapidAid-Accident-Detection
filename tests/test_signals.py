import pandas as pd
import pytest
import cv2
import numpy as np

from src.engine.signal_extractor import extract_optical_flow_signals, extract_tracking_signals


def _detection(frame_index: int, track_id: int, center_x: float) -> dict:
    return {
        "frame_index": frame_index,
        "track_id": track_id,
        "x1": center_x - 5,
        "y1": 10,
        "x2": center_x + 5,
        "y2": 20,
    }


def test_extract_tracking_signals_computes_pixel_speed_and_acceleration():
    detections = pd.DataFrame([
        _detection(0, 1, 0),
        _detection(1, 1, 10),
        _detection(2, 1, 30),
    ])

    signals = extract_tracking_signals(detections, fps=1, frame_count=3)

    assert signals["vehicle_count"].tolist() == [1, 1, 1]
    assert signals.loc[1, "median_pixel_speed_per_second"] == pytest.approx(10)
    assert signals.loc[2, "median_pixel_speed_per_second"] == pytest.approx(20)
    assert signals.loc[2, "median_pixel_acceleration_per_second2"] == pytest.approx(10)


def test_extract_tracking_signals_keeps_empty_sampled_frames():
    detections = pd.DataFrame([_detection(0, 1, 5), _detection(2, 1, 15)])

    signals = extract_tracking_signals(detections, fps=2, frame_count=5, sample_every=2)

    assert signals["frame_index"].tolist() == [0, 2, 4]
    assert signals["vehicle_count"].tolist() == [1, 1, 0]
    assert signals.loc[1, "median_pixel_speed_per_second"] == pytest.approx(10)


def test_extract_tracking_signals_rejects_invalid_timing():
    with pytest.raises(ValueError, match="fps and sample_every"):
        extract_tracking_signals(pd.DataFrame(), fps=0, frame_count=10)


def test_extract_optical_flow_detects_motion_and_static_frames(tmp_path):
    path = tmp_path / "moving_square.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 96))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((96, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + index * 4, 30), (35 + index * 4, 55), (255, 255, 255), -1)
        writer.write(frame)
    for _ in range(4):
        writer.write(frame)
    writer.release()

    signals = extract_optical_flow_signals(str(path), fps=10, frame_count=12, sample_every=1)

    assert len(signals) == 11
    assert signals.loc[1:5, "mean_flow_px_per_sample"].mean() > signals.loc[8:10, "mean_flow_px_per_sample"].mean()
    assert signals["high_motion_pixel_fraction"].between(0, 1).all()


def test_extract_optical_flow_rejects_invalid_timing():
    with pytest.raises(ValueError, match="fps, sample_every"):
        extract_optical_flow_signals("missing.mp4", fps=0, frame_count=10)
