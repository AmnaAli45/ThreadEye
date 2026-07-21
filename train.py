"""

Trains a YOLOv8 SEGMENTATION model on our prepared fabric defect dataset.

Settings below are tuned for a small (4GB) GPU like the Quadro P600 -
smaller image size and batch size so we don't run out of GPU memory.
If you have a bigger GPU later, you can increase these numbers.

To change any setting, just edit the values in the "SETTINGS" section
below and re-run the script - no need to type anything extra.

Usage:
    python src/train.py
"""

from pathlib import Path
from ultralytics import YOLO


EPOCHS = 100            # how many times the model goes through the full dataset
IMAGE_SIZE = 416        # training image size - smaller uses less GPU memory
BATCH_SIZE = 4          # how many images processed at once - smaller uses less GPU memory
PATIENCE = 20           # stop early if no improvement for this many epochs
BASE_MODEL = "yolov8n-seg.pt"  # smallest YOLO segmentation model - good fit for 4GB GPU

RUN_NAME = "threadeye_seg"      # name for this training run (used for the output folder)


# Path

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"


def main():
    
    # Make sure the dataset config file actually exists before
    
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found. Make sure data.yaml is in the project root.")
        return

    print("Starting training with these settings:")
    print(f"  Model:      {BASE_MODEL}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Patience:   {PATIENCE}")
    print(f"  Data config: {DATA_YAML}")
    print()

    
    # Load a pretrained YOLO segmentation model.
    # "yolov8n-seg.pt" is downloaded automatically the first time you
    # run this - it already knows general shapes/edges from training
    # on millions of images, we're just fine-tuning it on our fabric data.
    
    model = YOLO(BASE_MODEL)

    
    # STEP 3: Train the model.
    # device=0 tells it to use your GPU.
    # If GPU memory runs out, lower BATCH_SIZE or IMAGE_SIZE above and re-run.
    
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=0,                 # use the GPU
        project="runs",           # where results get saved
        name=RUN_NAME,            # subfolder name for this run
        exist_ok=True,            # allow re-running without error
    )

    print()
    print("Training complete!")
    print(f"Best model saved to: runs/{RUN_NAME}/weights/best.pt")
    print(f"Check runs/{RUN_NAME}/ for training graphs and results.")


if __name__ == "__main__":
    main()