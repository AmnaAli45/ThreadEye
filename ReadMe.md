# 🧵 ThreadEye

**AI-assisted fabric defect detection for textile mills**

ThreadEye is a computer vision prototype that scans fabric images for visible
defects — broken yarn, holes, stains, and weaving irregularities — using a
custom-trained YOLOv8 segmentation model. It's built for small and medium
textile mills in Faisalabad, Pakistan, where automated quality control tools
remain unaffordable or unavailable.

---

## Status

**Early-stage prototype.** Trained on the public AITEX fabric defect dataset
(244 source images) as a proof of concept. Detection accuracy is currently
limited and will improve substantially once fine-tuned on real fabric samples
from a pilot mill. This project is not yet production-ready.

---

## Features

- Detects and localizes fabric defects using image segmentation (not just
  bounding boxes — traces the actual defect shape)
- Simple web interface (Streamlit) for uploading a fabric image and viewing
  results instantly
- Adjustable sensitivity threshold to balance missed defects vs. false alarms
- Runs locally on a standard GPU — no cloud costs, no paid APIs

---

## Tech Stack

| Component | Tool |
|---|---|
| Object detection / segmentation | YOLOv8n-seg (Ultralytics) |
| Training | PyTorch (CUDA) |
| Image processing | OpenCV |
| Demo interface | Streamlit |
| Dataset | AITEX Fabric Image Database |

---

## Project Structure

```
ThreadEye/
├── data/
│   ├── raw/                  # Original AITEX images and masks (not tracked in git)
│   │   ├── Defect_images/
│   │   ├── Mask_images/
│   │   └── NODefect_images/
│   ├── processed/            # Tiled images + YOLO-format labels
│   └── dataset/              # Train/val/test split, ready for training
├── models/                   # Trained model weights (.pt)
├── runs/                     # Training logs, graphs, checkpoints (auto-generated)
├── outputs/                  # Saved prediction results
├── tile_dataset.py           # Converts masks to YOLO labels and tiles long images
├── split_dataset.py          # Splits processed data into train/val/test
├── train.py                  # Trains the YOLOv8-seg model
├── evaluate.py                # Evaluates the trained model on the test set
├── predict.py                # Runs the model on a single image (CLI)
├── app.py                    # Streamlit demo web app
├── data.yaml                 # Dataset configuration for YOLO
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create a virtual environment
```bash
git clone <repo-url>
cd ThreadEye
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

For GPU training (recommended), install the CUDA build of PyTorch matching
your GPU driver:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Add the dataset
Download the [AITEX Fabric Image Database](https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database)
and place it under:
```
data/raw/Defect_images/
data/raw/Mask_images/
data/raw/NODefect_images/
```

---

## Usage

### Prepare the dataset
```bash
python tile_dataset.py
python split_dataset.py
```

### Train the model
```bash
python train.py
```
Trained weights are saved to `runs/threadeye_seg/weights/best.pt`. Copy the
best checkpoint to the `models/` folder for use in the app:
```bash
copy runs\threadeye_seg\weights\best.pt models\threadeye_v1.pt
```

### Evaluate on the test set
```bash
python evaluate.py
```

### Run inference on a single image
```bash
python predict.py path/to/image.png
```

### Launch the demo app
```bash
streamlit run app.py
```

---

## Known Limitations

- Trained on a small, public dataset — not yet validated on real mill fabric
- Recall and precision are still low; some defects are missed and occasional
  false positives occur
- Only detects a single generic "defect" class (no defect-type classification yet)
- Tuned for a low-memory GPU (4GB); larger images/batches need more VRAM

---

## Roadmap

- [x] Data pipeline: mask-to-YOLO conversion, tiling for long fabric images
- [x] Baseline YOLOv8-seg model trained on AITEX
- [x] Streamlit demo interface
- [ ] Pilot with real mill fabric samples
- [ ] Fine-tune on real data, improve precision/recall
- [ ] Defect-type classification (holes, stains, broken yarn, etc.)
- [ ] Deployment for continuous/live inspection

---

## Dataset Credit

This project uses the **AITEX Fabric Image Database**, a public research
dataset for automatic fabric defect detection.

---

## License

Not yet determined — internal prototype.