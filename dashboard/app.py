from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import analyze_video, inspect_video


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "phase1"

st.set_page_config(page_title="RapidAid | Video Review", page_icon="🚦", layout="wide")
st.title("RapidAid")
st.caption("Local CCTV video review · Phase 1 vehicle tracking baseline")
st.info(
    "This phase finds and tracks visible cars, motorcycles, buses, and trucks. "
    "It does not yet decide whether a crash occurred or determine severity."
)

uploaded = st.file_uploader("Choose a traffic video", type=["mp4", "avi", "mov", "mkv"])
sample_every = st.select_slider("Analyze every Nth frame", options=[1, 2, 3, 5], value=2)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    job_id = uuid4().hex[:10]
    input_path = OUTPUTS / f"input_{job_id}{suffix}"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(uploaded.getvalue())
    try:
        info = inspect_video(input_path)
        a, b, c, d = st.columns(4)
        a.metric("Resolution", f"{info.width} × {info.height}")
        b.metric("Frame rate", f"{info.fps:.2f} FPS")
        c.metric("Duration", f"{info.duration_seconds:.1f} s")
        d.metric("Frames", f"{info.frame_count:,}")
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if st.button("Analyze vehicles", type="primary"):
        output_video = OUTPUTS / f"annotated_{job_id}.mp4"
        progress = st.progress(0, text="Loading detector and tracking vehicles…")
        try:
            with st.spinner("Processing video locally…"):
                detections, info, device = analyze_video(
                    input_path,
                    output_video,
                    sample_every=sample_every,
                    progress=lambda value: progress.progress(value, text=f"Tracking vehicles… {value:.0%}"),
                )
            st.session_state["phase1_result"] = {
                "job_id": job_id,
                "detections": detections,
                "video": str(output_video),
                "device": device,
            }
        except Exception as exc:
            st.exception(exc)

result = st.session_state.get("phase1_result")
if result:
    detections: pd.DataFrame = result["detections"]
    if detections.empty:
        st.warning("No COCO vehicle detections were returned. This can happen with distant, occluded, or low quality vehicles.")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Annotated review video")
            st.video(result["video"])
        with right:
            st.subheader("Run summary")
            st.metric("Vehicle observations", f"{len(detections):,}")
            st.metric("Unique track IDs", int(detections["track_id"].nunique()))
            st.metric("Inference device", "CUDA GPU" if str(result["device"]) == "0" else "CPU")

        track_counts = detections.groupby("time_seconds")["track_id"].nunique().rename("vehicle tracks").reset_index()
        st.subheader("Tracked vehicles over time")
        st.plotly_chart(px.line(track_counts, x="time_seconds", y="vehicle tracks", markers=True), use_container_width=True)
        st.download_button(
            "Download detections CSV",
            detections.to_csv(index=False).encode("utf-8"),
            file_name=f"rapidaid_detections_{result['job_id']}.csv",
            mime="text/csv",
        )
        with st.expander("Detection records"):
            st.dataframe(detections, use_container_width=True, hide_index=True)
