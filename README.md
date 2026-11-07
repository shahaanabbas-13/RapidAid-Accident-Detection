# RapidAid

RapidAid is a research prototype for CCTV video review and accident event detection. It is being rebuilt in phases. **Phase 1 is complete; Phase 2 has started with track-derived motion measurements.**

## Design

```text
CCTV / uploaded video → edge device (future) → cloud services (future)
                                  │
                     current local web prototype
```

The first prototype runs locally in a browser-backed web app. It uses a pretrained vehicle detector and tracker to produce evidence for review. A detector trained on ordinary COCO classes does **not** detect crashes by itself; accident detection and severity claims will only be added after a labeled evaluation set and measurable validation exist. The earlier report's 8/9 result is treated as a historical claim, not a verified benchmark.

## Phases

1. **Video baseline (implemented):** upload CCTV clips, inspect metadata, detect/track vehicles, review annotated video, and export detections.
2. **Physics signals (in progress):** track-derived image-plane speed and acceleration, plus whole-frame optical-flow magnitude, are implemented and viewable/exportable. Next: compare these traces across representative normal and incident footage, then add geometry and track-loss signals; candidate-event timing and clip/storyboard extraction follow.
3. **Evaluation and calibration:** create a manifest with video-level labels and event timestamps; report recall, precision, false alarms per hour, timing error, and runtime; train fusion only with enough independent videos.
4. **Semantic review:** optional local/API narration as advisory evidence, with secrets kept outside source control.
5. **Case workflow:** incident review, FIR-style records, maps and simulated notifications. No real emergency message is sent by the research prototype.
6. **Edge/cloud design:** profile and optimize a validated model for the target hardware before attempting deployment.

## Environment (Windows, Python 3.12)

This machine has an RTX 4050 Laptop GPU (6 GB VRAM). Use a fresh `.venv`; the existing `venv` in this checkout points to a Python installation from another machine and is not usable.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# NVIDIA driver reports CUDA 12.8; install the matching official PyTorch wheel.
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Then run `streamlit run dashboard/app.py`. The first model run downloads pretrained weights. Keep test videos in a local, untracked data folder; do not commit footage or model weights.

## Data and training guidance

Start with a small, legally usable CCTV dataset for end-to-end experiments. ACCIDENT is surveillance-focused and offers real plus synthetic clips with event localization/type annotations; keep its official license and split rules with the data. AI City Challenge Track 3 offers broad traffic-anomaly annotations, but source videos must be retrieved under each source's license. Dashcam datasets such as DADA-2000 and DoTA are useful for auxiliary research but are a different camera domain and should not stand in for fixed CCTV validation. Split by source video/camera/location, never by adjacent frames, to avoid leakage. Begin with pretrained inference and signal analysis; do not fine-tune on the nine-video legacy set.

## Current scope

This is a decision-support prototype. It is not validated for dispatch, severity decisions, or unattended emergency response. For now, any incident actions are simulated and require human review.
