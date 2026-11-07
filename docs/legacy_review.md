# Legacy RapidAid review

Reviewed the public repository linked for this rebuild on 2026-10-25. The checkout has two overlapping application generations:

- The root `main_pipeline.py` orchestrates the video path in `orchestration/`: track-centric RapidAid analysis, event clip extraction, storyboard/frame selection, narration and Groq reasoning, consensus tiering, and report artifacts.
- `RapidAid-Accident-Detection-System/` contains an earlier frame/image pipeline and a later `TrackCentricProcessor`. Its track-centric engine combines YOLO detections, custom persistent tracks, velocity and trajectory scores, Farneback optical flow, disappearance reasoning, confidence fusion, and a causal state gate.
- `src/` is a separate image-to-report Streamlit application with hospital and police pages. It uses an Ollama vision model and has PDF/report workflow code. The video orchestrator is a separate command-line entry point.

## Rebuild decisions

1. Keep one video-first entry point and a typed interface between video decoding, detection/tracking, signals, events, and reporting.
2. Separate observed measurements from learned probabilities. COCO vehicle confidence, track stability, image-plane motion, and optical-flow magnitude are not calibrated accident probabilities.
3. Treat optical-flow bursts and track disappearance as candidate evidence only: camera shake, compression, occlusion, shadows, and normal exits can trigger them.
4. Avoid the legacy `physics-overwhelming` rule as a dispatch authorization. A fixed threshold cannot guarantee zero false negatives or safe severity assignment; use it only as a review aid until held-out CCTV validation supports a policy.
5. Keep severity, victim counts, FIR creation, and notifications out of the first detector milestone. Simulate downstream actions and require a human confirmation step.
6. The report describes logistic-regression fusion, while the reviewed `ConfidenceFusion` implementation exposes configurable fixed weights. Rebuild fusion from an explicit, versioned training/evaluation process rather than assuming the report's model exists.

## Evaluation and training safeguards

- Use a manifest with source, license, camera/domain, video-level split, event interval, accident type, severity, and annotation provenance.
- Split by source video and, when possible, by camera/location. Never let adjacent frames or clips from one original video cross train and test partitions.
- Report event recall, precision, false alarms per camera-hour, event-time error, and runtime. Break results out by real/synthetic source, orientation, lighting, traffic density, and event class.
- Begin with pretrained detection and simple per-video analysis. Fine-tune only after validating labels and representative coverage; nine clips are an evaluation smoke set, not a sufficient training corpus.
- Keep downloaded videos, annotation files with restricted terms, model weights, and API secrets out of Git.

## Starter video sources

- [ACCIDENT benchmark](https://accidentbench.github.io/): CCTV surveillance clips, with real and synthetic subsets and accident time/location/type tasks. Review the benchmark and source-specific terms before downloading.
- [AI City Challenge 2026 Track 3](https://www.aicitychallenge.org/2026-track3/): traffic anomaly annotations; source videos are fetched from their original datasets and inherit those datasets' terms.
- [DoTA](https://arxiv.org/abs/2004.03044) and [DADA-2000](https://arxiv.org/abs/1904.12634): useful auxiliary driving-event research data, but primarily ego/dashcam views and therefore domain-shifted from fixed CCTV.
