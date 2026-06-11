"""
BGNet Dataset Builder
=====================
Real blood cell images download karne ka script
Sources: Kaggle BCCD Dataset + manual augmentation

Run: python download_dataset.py

Ye script automatically dataset/ folder banata hai
with 8 class folders (O+, A+, B+, AB+, O-, A-, B-, AB-)
"""

import os
import shutil
import urllib.request
import zipfile
import random
from pathlib import Path

CLASSES = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
DATASET_DIR = Path("../dataset")


def create_folder_structure():
    """Create 8-class dataset folder structure"""
    print("Creating dataset folder structure...")
    for cls in CLASSES:
        folder = DATASET_DIR / cls
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  Created: dataset/{cls}/")
    print("Done!\n")


def print_download_instructions():
    """Print manual download instructions for real dataset"""
    print("=" * 60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("OPTION 1: Kaggle (Best - Real Blood Smear Images)")
    print("-" * 40)
    print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/blood-cells")
    print("2. Download dataset2-master.zip")
    print("3. Extract and copy images to:")
    for cls in CLASSES:
        print(f"   dataset/{cls}/  (put blood smear images here)")
    print()
    print("OPTION 2: GitHub BCCD Dataset (Free)")
    print("-" * 40)
    print("git clone https://github.com/Shenggan/BCCD_Dataset.git")
    print("Then copy images from BCCD_Dataset/BCCD/JPEGImages/ to dataset folders")
    print()
    print("OPTION 3: Roboflow (Easy)")
    print("-" * 40)
    print("1. Go to: https://roboflow.com/")
    print("2. Search: 'blood cell detection'")
    print("3. Download free dataset")
    print()
    print("MINIMUM IMAGES NEEDED per class: 50 (recommended: 150+)")
    print()
    print("=" * 60)


def augment_existing_images():
    """
    Agar kuch images already hain, unhe augment karo
    taaki dataset bada ho jaye
    Uses: flip, rotate, brightness changes
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Install opencv first: pip install opencv-python")
        return

    print("Augmenting existing images...")
    total_added = 0

    for cls in CLASSES:
        folder = DATASET_DIR / cls
        if not folder.exists():
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if len(images) == 0:
            print(f"  {cls}: No images found, skipping")
            continue

        print(f"  {cls}: Found {len(images)} images, augmenting...")
        added = 0

        for img_path in images[:20]:  # Max 20 original images augment karo
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            augmentations = []

            # Flip horizontal
            augmentations.append(cv2.flip(img, 1))

            # Flip vertical
            augmentations.append(cv2.flip(img, 0))

            # Rotate 90
            augmentations.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

            # Rotate 270
            augmentations.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

            # Brightness +30
            bright = cv2.add(img, np.ones_like(img) * 30)
            augmentations.append(bright)

            # Brightness -30
            dark = cv2.subtract(img, np.ones_like(img) * 30)
            augmentations.append(dark)

            # Gaussian noise
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            noisy = cv2.add(img, noise)
            augmentations.append(noisy)

            for i, aug_img in enumerate(augmentations):
                base_name = img_path.stem
                save_path = folder / f"{base_name}_aug{i}.jpg"
                if not save_path.exists():
                    cv2.imwrite(str(save_path), aug_img)
                    added += 1

        total_added += added
        print(f"    → Added {added} augmented images")

    print(f"\nTotal augmented images added: {total_added}")


def show_dataset_stats():
    """Show current dataset statistics"""
    print("\nCURRENT DATASET STATS:")
    print("-" * 30)
    total = 0
    for cls in CLASSES:
        folder = DATASET_DIR / cls
        if folder.exists():
            count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
            total += count
            status = "✓" if count >= 50 else "⚠ (need more)"
            print(f"  {cls:5s}: {count:4d} images {status}")
        else:
            print(f"  {cls:5s}:    0 images ✗ (folder missing)")
    print(f"\n  TOTAL: {total} images")
    print(f"  STATUS: {'Ready to train!' if total >= 400 else 'Need more images (min 400 total)'}")


if __name__ == "__main__":
    create_folder_structure()
    print_download_instructions()
    augment_existing_images()
    show_dataset_stats()
