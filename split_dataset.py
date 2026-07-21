"""
What this script does:
    Takes all images + labels from data/processed/images and
    data/processed/labels, and splits them into train/val/test folders
    in the structure YOLO expects for training:

    data/dataset/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/

Usage:
    python src/split_dataset.py
"""

import random
import shutil
from pathlib import Path

# -------------------------------------------------------------- Path Setting ---------------------------------------------------------------------

# Find the project's root folder (2 levels up from this script, since this
# script lives inside src/)
ROOT = Path(__file__).resolve().parent

# Where our images + labels currently live (from the data_prep step)
SOURCE_IMAGES = ROOT / "data" / "processed" / "images"
SOURCE_LABELS = ROOT / "data" / "processed" / "labels"

# Where we want the final, split dataset to live
DATASET_DIR = ROOT / "data" / "dataset"

# How we want to split the data (must add up to 1.0)
TRAIN_RATIO = 0.70 # 70 % data is used for training
VAL_RATIO = 0.20 # 20 % data is for validation
TEST_RATIO = 0.10 #10% data is for testing

# Setting a "seed" makes the random shuffle reproducible - meaning if you
# run this script again, you'll get the EXACT same split every time,
# instead of a different random split each run. This is good practice.
RANDOM_SEED = 42


def get_all_image_files():
    """
    STEP 2: Look inside the images folder and make a list of every
    image file we find there.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    image_files = []
    for file_path in SOURCE_IMAGES.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_files.append(file_path)

    return image_files


def split_file_list(all_files):
    """
    STEP 3: Randomly shuffle the list of files, then cut it into
    3 pieces: train, val, test - based on our ratios above.
    """
    # Shuffle so the split isn't biased (e.g. all "no-defect" images
    # accidentally ending up in one split)
    random.seed(RANDOM_SEED)
    shuffled_files = all_files.copy()
    random.shuffle(shuffled_files)

    total = len(shuffled_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = shuffled_files[:train_end]
    val_files = shuffled_files[train_end:val_end]
    test_files = shuffled_files[val_end:]

    return train_files, val_files, test_files


def copy_files_to_split(image_files, split_name):
    """
    STEP 4: For a given list of images (e.g. train_files), copy each
    image AND its matching label file into the correct split folder.
    """
    # Create the destination folders if they don't already exist
    dest_images_dir = DATASET_DIR / "images" / split_name
    dest_labels_dir = DATASET_DIR / "labels" / split_name
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    for image_path in image_files:
        # Copy the image itself
        shutil.copy2(image_path, dest_images_dir / image_path.name)

        # Find and copy the matching label file (same name, .txt extension)
        label_name = image_path.stem + ".txt"
        source_label_path = SOURCE_LABELS / label_name

        if source_label_path.exists():
            shutil.copy2(source_label_path, dest_labels_dir / label_name)
        else:
            # This shouldn't normally happen if data_prep.py ran correctly,
            # but we handle it gracefully just in case.
            print(f"  WARNING: No label found for {image_path.name}, creating empty label.")
            (dest_labels_dir / label_name).write_text("")

        copied_count += 1

    return copied_count


def main():
    print("Starting dataset split...")
    print(f"Source images: {SOURCE_IMAGES}")
    print(f"Source labels: {SOURCE_LABELS}")
    print(f"Output:        {DATASET_DIR}")
    print()

    # Safety check: make sure the source folder actually exists and has images
    if not SOURCE_IMAGES.exists():
        print(f"ERROR: {SOURCE_IMAGES} does not exist. Did you run data_prep.py first?")
        return

    all_images = get_all_image_files()

    if len(all_images) == 0:
        print("ERROR: No images found. Did you run data_prep.py first?")
        return

    print(f"Found {len(all_images)} total images.")

    # Split the list into train/val/test
    train_files, val_files, test_files = split_file_list(all_images)

    print(f"Splitting into: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    print()

    # Copy each split into its own folder
    print("Copying train files...")
    train_count = copy_files_to_split(train_files, "train")

    print("Copying val files...")
    val_count = copy_files_to_split(val_files, "val")

    print("Copying test files...")
    test_count = copy_files_to_split(test_files, "test")

    print()
    print("-" * 50)
    print(f"Train: {train_count} images -> {DATASET_DIR / 'images' / 'train'}")
    print(f"Val:   {val_count} images -> {DATASET_DIR / 'images' / 'val'}")
    print(f"Test:  {test_count} images -> {DATASET_DIR / 'images' / 'test'}")
    print("Done! Dataset is ready for training.")


if __name__ == "__main__":
    main()