import os
import cv2
import argparse
from tqdm import tqdm
from src.config import DATASETS_DIR

def clean_dataset(dataset_name):
    target_dir = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.exists(target_dir):
        print(f"[WARNING] Directory {target_dir} does not exist. Skipping cleaning.")
        return

    print(f"[INFO] Scanning and cleaning corrupted images in {dataset_name}...")
    removed_count = 0
    valid_count = 0

    for root, _, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        os.remove(file_path)
                        removed_count += 1
                    else:
                        valid_count += 1
                except Exception:
                    os.remove(file_path)
                    removed_count += 1

    print(f"[INFO] {dataset_name} cleaning complete. Valid images: {valid_count}, Removed corrupted images: {removed_count}")

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing and Cleaning Utility")
    parser.add_argument("--dataset", type=str, default="all", choices=["plantvillage", "plantdoc", "all"], help="Dataset to clean")
    args = parser.parse_args()

    if args.dataset in ["plantvillage", "all"]:
        clean_dataset("plantvillage")
    if args.dataset in ["plantdoc", "all"]:
        clean_dataset("plantdoc")

if __name__ == "__main__":
    main()
