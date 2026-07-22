"""

Evaluates a trained model on the TEST set (images it has never seen
during training or validation) and prints performance metrics.

Usage:
    python src/evaluate.py
"""

from pathlib import Path
from ultralytics import YOLO


# SETTINGS - change these if needed

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "threadeye_v1.pt"   # the trained model to evaluate
DATA_YAML = ROOT / "data.yaml"


def main():
    
    # Safety checks
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Make sure you copied best.pt there after training.")
        return

    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        return

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    
    #  Run evaluation on the TEST split specifically.
    # split="test" tells it to use the test folder, not val.
   
    print("Running evaluation on test set...")
    metrics = model.val(data=str(DATA_YAML), split="test")

    
    # Print the key numbers in plain language
    
    print()
    print("-" * 50)
    print("EVALUATION RESULTS (on test set - images the model never saw)")
    print("-" * 50)
    print(f"mAP50 (mask):     {metrics.seg.map50:.3f}   <- overall accuracy at 50% overlap threshold")
    print(f"mAP50-95 (mask):  {metrics.seg.map:.3f}   <- stricter overall accuracy (harder metric)")
    print(f"Precision:        {metrics.seg.mp:.3f}   <- of predicted defects, how many were correct")
    print(f"Recall:           {metrics.seg.mr:.3f}   <- of actual defects, how many were found")
    print()
    print("Rule of thumb: closer to 1.0 = better. For a first model on a")
    print("small dataset, anything above 0.5 is a reasonable starting point.")


if __name__ == "__main__":
    main()