import os
"""
This script is used to rename the images in the dataset.
It is a custom script that is used to rename the images in the dataset.
Author: @https://github.com/JeanViaunel
Date: 2025-05-25
Version: 1.0.0

Usage:
python renaming.py

Installation:
pip install -r requirements.txt
"""

# Root folder where subclass folders exist
root_dir = "Final-Competition-2025"  # change if needed

# Mapping from folder name to class prefix
class_prefixes = {
    "Class A": "image",
    "Class B": "image1",
    "Class C": "image2"
}

# Counter to generate sequential IDs per class
counters = {
    "Class A": 0,
    "Class B": 100,
    "Class C": 200
}

for class_name, prefix in class_prefixes.items():
    class_path = os.path.join(root_dir, class_name)
    if not os.path.exists(class_path):
        print(f"⚠️ Folder not found: {class_path}")
        continue

    for fname in sorted(os.listdir(class_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Generate new name
        ext = os.path.splitext(fname)[1]
        new_index = counters[class_name]
        new_name = f"{prefix}{new_index:03d}{ext}"

        src_path = os.path.join(class_path, fname)
        dst_path = os.path.join(class_path, new_name)

        os.rename(src_path, dst_path)
        print(f"Renamed: {fname} -> {new_name}")

        counters[class_name] += 1
