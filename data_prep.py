"""

ThreadEye - Fabric Defect Detection (Segmentation version)
Converts AITEX-style mask images into YOLO-SEGMENTATION format labels
(polygon points instead of bounding boxes).

Expected input structure (data/raw/):
    data/raw/Defect_images/      -> fabric images WITH defects
    data/raw/Mask_images/        -> corresponding B&W masks (white = defect area)
    data/raw/NODefect_images/    -> fabric images WITHOUT defects

Output (data/processed/):
    data/processed/images/       -> all images copied here (defect + no-defect)
    data/processed/labels/       -> YOLO-seg .txt label files (polygon points)

Usage:
    python src/data_prep_seg.py
    python src/data_prep_seg.py --min-area 4 --epsilon-factor 0.005
"""

import argparse
import shutil #  this library is used to copy the files
from pathlib import Path

import cv2

# ------------------------------------------------------------- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root (ThreadEye/)
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DEFECT_IMG_DIR = RAW_DIR / "Defect_images"
MASK_DIR = RAW_DIR / "Mask_images"
NODEFECT_IMG_DIR = RAW_DIR / "NODefect_images"

OUT_IMG_DIR = PROCESSED_DIR / "images"
OUT_LABEL_DIR = PROCESSED_DIR / "labels"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Minimum points a polygon needs to be a valid shape for YOLO-seg
MIN_POLYGON_POINTS = 3

# ------------------ Helper Function : to avoid file name errors ----------------------------------------------------------------------
def find_file_case_insensitive(folder: Path, stem: str) -> Path | None:
    """Find a file in `folder` matching `stem` regardless of extension/case."""
    if not folder.exists():
        return None
    for f in folder.iterdir():
        if f.is_file() and f.stem.lower() == stem.lower() and f.suffix.lower() in VALID_EXTENSIONS:
            return f
    return None

# ---------------------------------------- Converting Mask Images to YOLO Format ------------------------------------------------------------
def mask_to_yolo_seg_lines(
    mask_path: Path, class_id: int, min_area: int, epsilon_factor: float
) -> list[str]:
    """
    Read a binary mask image and convert white defect regions into
    YOLO-SEGMENTATION format label lines:
        "class_id x1 y1 x2 y2 ... xn yn"  (all normalized 0-1)

    epsilon_factor controls how much the polygon is simplified:
    smaller = more points/detail, larger = fewer points/simpler shape.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"WARNING: Could not read mask: {mask_path}")
        return []

    h_img, w_img = mask.shape

    # Threshold to ensure a clean binary mask
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find out the outline of white shapes

    lines = []
   
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Simplify the contour so we don't store hundreds of near-duplicate points
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < MIN_POLYGON_POINTS:
            continue

        # Flatten points and normalize
        coords = []
        for point in approx:
            x, y = point[0]
            coords.append(f"{x / w_img:.6f}")
            coords.append(f"{y / h_img:.6f}")

        lines.append(f"{class_id} " + " ".join(coords))

    return lines


# Ye function sab defect images pe loop chalata hai aur har ek ke liye:

# Uska matching mask dhoondta hai
# Mask ko polygon lines mein convert karta hai (upar wala function call kar ke)
# Image ko data/processed/images/ mein copy karta hai
# Polygon lines ko .txt file mein save karta hai

# Agar koi mask nahi milta, wo image skip ho jati hai (aur counter mein record hota hai kitni skip hui).


def process_defect_images(class_id: int, min_area: int, epsilon_factor: float) -> tuple[int, int]:
    """Process all Defect_images + Mask_images pairs. Returns (processed, skipped)."""
    if not DEFECT_IMG_DIR.exists():
        print(f"ERROR: Defect images folder not found: {DEFECT_IMG_DIR}")
        return 0, 0

    processed, skipped = 0, 0

    for img_path in sorted(DEFECT_IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        mask_path = find_file_case_insensitive(MASK_DIR, img_path.stem)
        if mask_path is None:
            print(f"WARNING: No matching mask found for {img_path.name}, skipping.")
            skipped += 1
            continue

        lines = mask_to_yolo_seg_lines(mask_path, class_id, min_area, epsilon_factor)

        # Copy image to processed/images/
        dest_img = OUT_IMG_DIR / img_path.name
        shutil.copy2(img_path, dest_img)

        # Write label file (even if empty, for consistency)
        label_path = OUT_LABEL_DIR / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines))

        processed += 1

    return processed, skipped


# NODefect_images ki har image ko copy karta hai, aur uske liye ek empty .txt file banata hai (matlab: "is image mein koi defect nahi hai" YOLO ko ye batana zaroori hai, taake wo clean fabric bhi pehchanna seekhe).

def process_nodefect_images() -> int:
    """Copy no-defect images and create empty label files."""
    if not NODEFECT_IMG_DIR.exists():
        print(f"WARNING: NoDefect folder not found: {NODEFECT_IMG_DIR} (skipping)")
        return 0

    count = 0
    for img_path in sorted(NODEFECT_IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        dest_img = OUT_IMG_DIR / img_path.name
        shutil.copy2(img_path, dest_img)

        # Empty label file = "no objects in this image" for YOLO
        label_path = OUT_LABEL_DIR / (img_path.stem + ".txt")
        label_path.write_text("")

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert AITEX masks to YOLO-segmentation labels.")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID to assign to all defects (default: 0)")
    parser.add_argument("--min-area", type=int, default=4, help="Minimum contour area (pixels) to keep, filters noise (default: 4)")
    parser.add_argument("--epsilon-factor", type=float, default=0.005, help="Polygon simplification factor - smaller=more detail (default: 0.005)")
    args = parser.parse_args()

    print("Starting data preparation (SEGMENTATION format)...")
    print(f"Raw data dir: {RAW_DIR}")
    print(f"Output dir:   {PROCESSED_DIR}")

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    defect_processed, defect_skipped = process_defect_images(
        args.class_id, args.min_area, args.epsilon_factor
    )
    nodefect_processed = process_nodefect_images()

    print("-" * 50)
    print(f"Defect images processed:    {defect_processed}")
    print(f"Defect images skipped:      {defect_skipped} (no matching mask)")
    print(f"No-defect images processed: {nodefect_processed}")
    print(f"Total images in dataset:    {defect_processed + nodefect_processed}")
    print(f"Output images:  {OUT_IMG_DIR}")
    print(f"Output labels:  {OUT_LABEL_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()