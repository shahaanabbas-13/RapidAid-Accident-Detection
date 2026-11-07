import cv2
import numpy as np
import pytest

from src.pipeline import inspect_video


def test_inspect_video_reads_valid_clip(tmp_path):
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((48, 64, 3), dtype="uint8"))
    writer.release()

    info = inspect_video(path)

    assert (info.width, info.height, info.frame_count) == (64, 48, 20)
    assert info.fps == pytest.approx(10.0)
    assert info.duration_seconds == pytest.approx(2.0)
    assert info.frame_count_estimated is False


def test_inspect_video_rejects_invalid_file(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="Could not open"):
        inspect_video(path)
