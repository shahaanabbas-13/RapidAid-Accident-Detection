import torch
from ultralytics import YOLO

def main():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading YOLO model...")

    # Download a small pretrained model on the first run.
    model = YOLO("yolo11n.pt")

    print("\nRunning a test inference...")

    # Run a test prediction using the GPU if available.
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    print("Inference completed successfully!")
    print("Detected objects:", len(results[0].boxes))

    print("\nRapidAid computer vision setup is working.")

if __name__ == "__main__":
    main()