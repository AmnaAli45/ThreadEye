"""


WHY THIS SCRIPT EXISTS:
Our fabric images are extremely long and thin (4096 x 256 pixels).
When YOLO resizes these to a square training size (e.g. 416x416), the
width gets squeezed ~10x. Defects that were already thin (sometimes
just a few pixels wide) become nearly invisible after this squeeze,
so the model can never learn to see them.

THE FIX: instead of feeding YOLO the whole 4096-wide image, we cut
each image into smaller, more "square-ish" tiles (512 x 256 by
default). Each tile has a much gentler aspect ratio (2:1 instead of
16:1), so defects stay visible after resizing for training.

This script:
    1. Takes each Defect_images + Mask_images pair
    2. Cuts them into overlapping tiles
    3. For each tile, checks if it contains any defect (from the mask)
    4. Saves tiles that contain a defect, with a YOLO-segmentation label
    5. Also tiles NODefect_images (all clean, no defect) for balance,
       randomly sampling down to roughly match the number of defect tiles

Expected input structure (data/raw/):
    data/raw/Defect_images/
    data/raw/Mask_images/
    data/raw/NODefect_images/   (may contain sub-folders)

Output (data/processed_tiled/):
    data/processed_tiled/images/
    data/processed_tiled/labels/

Usage:
    python src/tile_dataset.py
"""

import random
import shutil
from pathlib import Path

import cv2


TILE_WIDTH = 512        # width of each tile in pixels
STRIDE = 384            # how far we move before cutting the next tile
                         # (smaller than TILE_WIDTH = overlapping tiles,
                         #  which helps avoid cutting a defect exactly at
                         #  a tile boundary)

MIN_DEFECT_AREA = 4     # minimum defect area (pixels) in a tile to count as "has defect"
MASK_THRESHOLD = 10     # same reasoning as before: AITEX masks use varying
                         # gray intensities, so we treat anything above this
                         # low value as "defect"
EPSILON_FACTOR = 0.01   # polygon simplification factor

NODEFECT_TILE_RATIO = 1.0  # how many no-defect tiles to keep, relative to
                            # the number of defect tiles found (1.0 = same amount)

RANDOM_SEED = 42


# Paths

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

DEFECT_IMG_DIR = RAW_DIR / "Defect_images"
MASK_DIR = RAW_DIR / "Mask_images"
NODEFECT_IMG_DIR = RAW_DIR / "NODefect_images"

OUT_IMG_DIR = OUT_DIR / "images"
OUT_LABEL_DIR = OUT_DIR / "labels"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIN_POLYGON_POINTS = 3


def find_mask_file(stem):
    """Find the mask file matching a defect image (AITEX uses '_mask' suffix)."""
    candidates = [f"{stem}_mask", stem]
    if not MASK_DIR.exists():
        return None
    for f in MASK_DIR.iterdir():
        if not f.is_file() or f.suffix.lower() not in VALID_EXTENSIONS:
            continue
        for candidate in candidates:
            if f.stem.lower() == candidate.lower():
                return f
    return None


def get_tile_x_positions(image_width, tile_width, stride):
    """
    Work out where each tile should start (the x pixel coordinate),
    making sure the last tile still fits inside the image (even if
    that means it overlaps more with the previous tile).
    """
    positions = list(range(0, image_width - tile_width + 1, stride))

    # Make sure we cover the very end of the image too
    last_possible_start = image_width - tile_width
    if len(positions) == 0 or positions[-1] != last_possible_start:
        if last_possible_start >= 0:
            positions.append(last_possible_start)

    return positions


def mask_tile_to_yolo_lines(mask_tile, class_id=0):
    """
    Same idea as our earlier mask_to_yolo_seg conversion, but applied
    to a single small tile instead of the whole image.
    """
    h_tile, w_tile = mask_tile.shape

    _, thresh = cv2.threshold(mask_tile, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_DEFECT_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = EPSILON_FACTOR * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < MIN_POLYGON_POINTS:
            continue

        coords = []
        for point in approx:
            x, y = point[0]
            coords.append(f"{x / w_tile:.6f}")
            coords.append(f"{y / h_tile:.6f}")

        lines.append(f"{class_id} " + " ".join(coords))

    return lines


def process_defect_images():
    """
    Tile every defect image + its mask, keep only the tiles that
    actually contain a defect, and save them with YOLO-seg labels.
    Returns the number of defect tiles created.
    """
    if not DEFECT_IMG_DIR.exists():
        print(f"ERROR: {DEFECT_IMG_DIR} not found.")
        return 0

    tile_count = 0
    source_images_processed = 0

    for img_path in sorted(DEFECT_IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        mask_path = find_mask_file(img_path.stem)
        if mask_path is None:
            print(f"WARNING: No mask for {img_path.name}, skipping.")
            continue

        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"WARNING: Could not read {img_path.name} or its mask, skipping.")
            continue

        img_h, img_w = mask.shape[:2]
        tile_height = img_h  # keep full height, we only tile along width

        x_positions = get_tile_x_positions(img_w, TILE_WIDTH, STRIDE)

        for tile_idx, x_start in enumerate(x_positions):
            x_end = x_start + TILE_WIDTH

            image_tile = image[:, x_start:x_end]
            mask_tile = mask[:, x_start:x_end]

            lines = mask_tile_to_yolo_lines(mask_tile)

            if len(lines) == 0:
                continue  # this tile has no defect, skip it here (handled separately)

            tile_name = f"{img_path.stem}_tile{tile_idx}.png"
            cv2.imwrite(str(OUT_IMG_DIR / tile_name), image_tile)

            label_path = OUT_LABEL_DIR / (Path(tile_name).stem + ".txt")
            label_path.write_text("\n".join(lines))

            tile_count += 1

        source_images_processed += 1

    print(f"Defect source images processed: {source_images_processed}")
    print(f"Defect tiles created (contain a defect): {tile_count}")
    return tile_count


def process_nodefect_images(target_tile_count):
    """
    Tile every no-defect image, then randomly keep roughly
    `target_tile_count` tiles so the dataset stays balanced.
    """
    if not NODEFECT_IMG_DIR.exists():
        print(f"WARNING: {NODEFECT_IMG_DIR} not found, skipping no-defect tiles.")
        return 0

    all_candidate_tiles = []  # list of (image_path, x_start, tile_height)

    for img_path in sorted(NODEFECT_IMG_DIR.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # We only need dimensions here, so read in grayscale for speed
        probe = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if probe is None:
            continue

        img_h, img_w = probe.shape[:2]
        x_positions = get_tile_x_positions(img_w, TILE_WIDTH, STRIDE)

        for x_start in x_positions:
            all_candidate_tiles.append((img_path, x_start))

    print(f"No-defect candidate tiles available: {len(all_candidate_tiles)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_candidate_tiles)

    keep_count = min(target_tile_count, len(all_candidate_tiles))
    selected_tiles = all_candidate_tiles[:keep_count]

    saved = 0
    for img_path, x_start in selected_tiles:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        x_end = x_start + TILE_WIDTH
        image_tile = image[:, x_start:x_end]

        tile_name = f"{img_path.stem}_nodefect_x{x_start}.png"
        cv2.imwrite(str(OUT_IMG_DIR / tile_name), image_tile)

        # Empty label = "no objects in this tile"
        label_path = OUT_LABEL_DIR / (Path(tile_name).stem + ".txt")
        label_path.write_text("")

        saved += 1

    print(f"No-defect tiles saved: {saved}")
    return saved


def main():
    print("Starting tiling process...")
    print(f"Tile size: {TILE_WIDTH}px wide (full height), stride: {STRIDE}px")
    print()

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    defect_tile_count = process_defect_images()

    target_nodefect = int(defect_tile_count * NODEFECT_TILE_RATIO)
    nodefect_tile_count = process_nodefect_images(target_nodefect)

    print()
    print("-" * 50)
    print(f"Total defect tiles:    {defect_tile_count}")
    print(f"Total no-defect tiles: {nodefect_tile_count}")
    print(f"Total tiles:           {defect_tile_count + nodefect_tile_count}")
    print(f"Output images: {OUT_IMG_DIR}")
    print(f"Output labels: {OUT_LABEL_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()