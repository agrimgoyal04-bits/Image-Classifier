import os
import csv
import shutil
from pathlib import Path

# ==== CONFIGURE THESE ====
CSV_PATH = "/Users/agrimgoyal/Desktop/img_cls_vit/data/processed_dataset.csv"           # your CSV file
IMAGES_ROOT = "/Users/agrimgoyal/Downloads/images"           # folder with 12k images
OUTPUT_DIR = "/Users/agrimgoyal/Desktop/DLP/data/pooled_images"         # folder to store extracted images
COLUMN = "image_path"                        # column name in the CSV
# ==========================


def main():
    csv_path = Path(CSV_PATH)
    images_root = Path(IMAGES_ROOT) 
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read relative paths from CSV (e.g., "images/team14_wrist_watch (1).jpg")
    image_paths = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row[COLUMN].strip()
            if not raw:
                continue
            # Only keep the filename (CSV contains paths like "images/filename.jpg")
            rel_path = Path(raw).name
            image_paths.add(Path(rel_path))

    print(f"Found {len(image_paths)} unique filenames in CSV.")

    copied = 0
    missing = 0

    # Copy images
    for rel_path in image_paths:
        # First, assume the CSV path is relative to IMAGES_ROOT
        src = images_root / rel_path.name

        if src.is_file():
            dest = output_dir / rel_path.name
            shutil.copy2(src, dest)
            copied += 1
            continue

        # Fall back: search anywhere under IMAGES_ROOT by filename
        found = list(images_root.rglob(rel_path))
        if found:
            src = found[0]
            dest = output_dir / rel_path.name
            shutil.copy2(src, dest)
            copied += 1
        else:
            print(f"Missing: {rel_path}")
            missing += 1

    print(f"\nCopied: {copied}")
    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()