import os
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

"""
This script is used to label the images in the dataset.
It is a custom script that is used to label the images in the dataset.
Author: @https://github.com/JeanViaunel
Date: 2025-05-25
Version: 1.0.0

Usage:
python labelme_custom.py

Installation:
pip install -r requirements.txt

Requirements:
- Python 3.7+
- pandas
- Pillow
- tkinter
- ttkthemes
"""

# === CONFIGURATION ===
base_dir = "Final-Competition-2025"
output_csv = "labels_labeled.csv"

TK_SILENCE_DEPRECATION=1

# Folder name → class label
class_map = {"Class A": 18, "Class B": 19, "Class C": 20}
damage_options = {
    "Class A": ["0 - Exposed rebar"],
    "Class B": ["3 - X and V-shaped cracks", "4 - Continuous diagonal cracks", "6 - Continuous vertical cracks", "8 - Continuous horizontal cracks"],
    "Class C": ["1 - No significant damage", "5 - Discontinuous diagonal cracks", "7 - Discontinuous vertical cracks", "9 - Discontinuous horizontal cracks", "10 - Small cracks"]
}

# === Load Existing Labels (if any) ===
if os.path.exists(output_csv):
    labeled_df = pd.read_csv(output_csv)
    labeled_ids = set(labeled_df["id"].astype(str))
else:
    labeled_df = pd.DataFrame()
    labeled_ids = set()

# === GUI SETUP FOR CLASS SELECTION ===
def start_labeling(selected_class):
    cls_label = class_map[selected_class]
    allowed_damages = damage_options[selected_class]
    folder_path = os.path.join(base_dir, selected_class)

    # Load images from selected class
    entries = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            image_id = os.path.splitext(fname)[0]
            if image_id in labeled_ids:
                continue  # Resume support: skip already labeled
            entries.append({
                "id": image_id,
                "class_label": cls_label,
                "filepath": os.path.join(folder_path, fname)
            })

    if not entries:
        messagebox.showinfo("Done", f"No unlabeled images found in {selected_class}.")
        root.destroy()
        return

    # === LAUNCH MAIN LABELING WINDOW ===
    label_window = tk.Toplevel(root)
    label_window.title(f"Labeling: {selected_class}")
    current = [0]  # mutable index
    checkbox_vars = []

    def load_image():
        for var in checkbox_vars:
            var.set(0)
        img_data = entries[current[0]]
        img = Image.open(img_data["filepath"]).resize((400, 400))
        photo = ImageTk.PhotoImage(img)
        panel.config(image=photo)
        panel.image = photo
        status_label.config(text=f"{current[0]+1}/{len(entries)} — {os.path.basename(img_data['filepath'])}")

    def save_and_next():
        selected = [label.split(" ")[0] for i, label in enumerate(allowed_damages) if checkbox_vars[i].get()]
        entries[current[0]]["damage_labels"] = ",".join(selected)
        current[0] += 1
        if current[0] < len(entries):
            load_image()
        else:
            save_to_csv()
            label_window.destroy()

    def save_to_csv():
        df_new = pd.DataFrame(entries)
        if not labeled_df.empty:
            df_final = pd.concat([labeled_df, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final[["id", "class_label", "damage_labels"]].to_csv(output_csv, index=False)
        print(f"✅ Labels saved to {output_csv}")

    # === Labeling UI ===
    panel = tk.Label(label_window)
    panel.pack()

    check_frame = ttk.Frame(label_window)
    check_frame.pack()

    checkbox_vars = [tk.IntVar() for _ in allowed_damages]
    for i, label in enumerate(allowed_damages):
        cb = ttk.Checkbutton(check_frame, text=label, variable=checkbox_vars[i])
        cb.grid(row=i//2, column=i%2, sticky="w")

    ttk.Button(label_window, text="Save & Next", command=save_and_next).pack(pady=10)
    status_label = tk.Label(label_window, text="")
    status_label.pack()

    load_image()

# === MAIN WINDOW FOR CLASS SELECTION ===
root = tk.Tk()
root.title("Select Class to Label")

ttk.Label(root, text="Select Class Folder:").pack(pady=10)

class_choice = tk.StringVar()
class_dropdown = ttk.Combobox(root, textvariable=class_choice, values=list(class_map.keys()), state="readonly")
class_dropdown.set("ClassA")  # default
class_dropdown.pack()

ttk.Button(root, text="Start Labeling", command=lambda: start_labeling(class_choice.get())).pack(pady=20)

root.mainloop()
