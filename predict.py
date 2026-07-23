"""

Takes a fabric image, runs our trained model on it, and saves a copy
of the image with any detected defects highlighted.

Usage:
    python predict.py path/to/some_fabric_image.png
"""

import sys
from pathlib import Path

from ultralytics import YOLO


# SETTINGS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "threadeye_v1.pt"
OUTPUT_DIR = ROOT / "outputs"
CONFIDENCE_THRESHOLD = 0.10  # only show predictions the model is at least this sure about


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running detection on: {image_path.name}")
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        exist_ok=True,
    )

    # Report what was found, in plain language
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0

    print()
    print("-" * 50)
    if num_detections == 0:
        print("No defects detected in this image.")
    else:
        print(f"{num_detections} potential defect(s) detected.")
        for i, box in enumerate(results[0].boxes):
            confidence = float(box.conf[0])
            print(f"  Defect {i+1}: {confidence*100:.1f}% confidence")

    print(f"Annotated image saved to: {OUTPUT_DIR / 'predictions'}")


if __name__ == "__main__":
    main()