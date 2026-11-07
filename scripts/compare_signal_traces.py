"""Run the current evidence signals on one normal and one labeled incident clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.signal_extractor import (
    extract_geometry_track_loss_signals,
    extract_optical_flow_signals,
    extract_tracking_signals,
)
from src.pipeline import analyze_video


FEATURES = [
    ("median_pixel_speed_per_second", "Track speed (pixels/s)"),
    ("mean_flow_px_per_sample", "Mean optical flow (pixels/sample)"),
    ("max_bbox_iou", "Maximum bbox IoU"),
    ("one_step_track_loss_count", "Tracks lost since prior sample"),
]


def analyze(path: Path, output_dir: Path, sample_every: int) -> tuple[pd.DataFrame, int, int]:
    info_path = output_dir / f"{path.stem}_annotated.mp4"
    detections, info, device = analyze_video(
        path,
        info_path,
        sample_every=sample_every,
    )
    signals = extract_tracking_signals(
        detections,
        fps=info.fps,
        frame_count=info.frame_count,
        sample_every=sample_every,
    )
    flow = extract_optical_flow_signals(
        str(path),
        fps=info.fps,
        frame_count=info.frame_count,
        sample_every=sample_every,
    )
    geometry = extract_geometry_track_loss_signals(
        detections,
        fps=info.fps,
        frame_count=info.frame_count,
        frame_width=info.width,
        frame_height=info.height,
        sample_every=sample_every,
    )
    for evidence in (flow, geometry):
        signals = signals.merge(evidence, on=["frame_index", "time_seconds"], how="left")
    signals.to_csv(output_dir / f"{path.stem}_signals.csv", index=False)
    return signals, len(detections), detections["track_id"].nunique() if not detections.empty else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True, help="Normal-traffic video")
    parser.add_argument("--incident", type=Path, required=True, help="Labeled incident video")
    parser.add_argument("--event-start", type=float, required=True, help="Manually reviewed event start time in seconds")
    parser.add_argument("--event-end", type=float, required=True, help="Manually reviewed event end time in seconds")
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "comparison")
    args = parser.parse_args()
    if args.event_start < 0 or args.event_end < args.event_start:
        parser.error("event interval must be non-negative and end at or after start")
    for clip in (args.normal, args.incident):
        if not clip.is_file():
            parser.error(f"video does not exist: {clip}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for name, path in (("Normal", args.normal), ("Incident", args.incident)):
        print(f"Analyzing {name.lower()} clip: {path}", flush=True)
        signals, observations, tracks = analyze(path, args.output_dir, args.sample_every)
        results[name] = signals
        print(f"  observations={observations}, track IDs={tracks}, samples={len(signals)}", flush=True)

    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(11, 10), sharex=False)
    for axis, (column, label) in zip(axes, FEATURES):
        for name, signals in results.items():
            if column in signals:
                axis.plot(signals["time_seconds"], signals[column], label=name, linewidth=1.5)
        if axis is axes[-1]:
            axis.axvspan(args.event_start, args.event_end, color="crimson", alpha=0.18, label="Reviewed event interval")
        else:
            axis.axvspan(args.event_start, args.event_end, color="crimson", alpha=0.12)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time from clip start (seconds)")
    fig.suptitle("RapidAid evidence-signal comparison")
    fig.tight_layout()
    chart_path = args.output_dir / "signal_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    normal_window = results["Normal"][results["Normal"]["time_seconds"].between(args.event_start, args.event_end)]
    incident_window = results["Incident"][results["Incident"]["time_seconds"].between(args.event_start, args.event_end)]
    summary_rows = []
    print(f"Aligned comparison for {args.event_start:.2f}–{args.event_end:.2f} s:")
    for column, label in FEATURES:
        normal_values = normal_window[column].dropna()
        incident_values = incident_window[column].dropna()
        if normal_values.empty or incident_values.empty:
            continue
        normal_mean = float(normal_values.mean())
        incident_mean = float(incident_values.mean())
        summary_rows.append({
            "signal": label,
            "event_start_seconds": args.event_start,
            "event_end_seconds": args.event_end,
            "normal_sample_count": len(normal_values),
            "incident_sample_count": len(incident_values),
            "normal_mean": normal_mean,
            "incident_mean": incident_mean,
            "difference": incident_mean - normal_mean,
        })
        print(f"  {label}: normal={normal_mean:.4g}, incident={incident_mean:.4g}")
    summary_path = args.output_dir / "comparison_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Comparison chart: {chart_path}")
    print(f"Aligned summary: {summary_path}")
    print(f"Per-video signal CSVs: {args.output_dir}")


if __name__ == "__main__":
    main()
