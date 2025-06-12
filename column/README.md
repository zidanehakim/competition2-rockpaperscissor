# Column Concrete Damage Detection Pipeline

A deep learning pipeline for detecting and classifying column concrete damage using YOLOv8 and ResNet50 models.

## System Overview

The pipeline has 4 main stages:
1. **YOLOv8 Detection** - Find damage regions in images
2. **Damage Classification** - Classify main images (classes 18, 19, 20)  
3. **Crack Classification** - Classify extracted regions (classes 3-9)
4. **Result Combination** - Merge predictions with mapping rules

## Prerequisites

```bash
pip install torch torchvision ultralytics opencv-python scikit-learn pandas numpy matplotlib seaborn tqdm pillow
```

## Step-by-Step Execution

### Step 1: Train YOLOv8 Model (if needed)

```bash
# Train YOLOv8 for damage detection
yolo train model=yolov8l.pt data=Yolo_v8_l_Detection/concrete_damage.yaml epochs=100 imgsz=1024 batch=24

# Output: runs/detect/train/weights/best.pt
```

### Step 2: Run YOLOv8 Detection

```bash
# Detect damage regions in test images
python "Yolo_v8_l_Detection/1. yolov8_test_script.py"

# Input: column/ (test images)
# Output: results_json/combined_results.json, detection_report.csv
```

### Step 3: Extract Bounding Boxes

```bash
# Extract detected regions as separate images
python "Yolo_v8_l_Detection/2. extract_boxes_script.py"

# Input: results_json/combined_results.json
# Output: extracted_boxes/ (sub-images), extracted_boxes_report.csv
```

### Step 4: Balance Dataset (Optional)

```bash
# Balance training dataset for better classification
python Single_Head_Classifier/Balancing_Dataset/simple_balancer.py

# Input: classification/ (training data organized by class folders)
# Output: classification_balanced/ (augmented balanced dataset)
```

### Step 5: Train Classification Models

```bash
# Train damage classifier (classes 18, 19, 20)
python Single_Head_Classifier/Damage_Single_Head/damage_classifier.py

# Train crack classifier (classes 3, 4, 5, 6, 7, 8, 9)
python Single_Head_Classifier/Crack_Single_Head/crack_classifier.py

# Outputs: best_damage_model.pth, best_crack_model.pth
```

### Step 6: Run Damage Classification

```bash
# Classify main images for damage severity
python Single_Head_Classifier/Damage_Single_Head/damage_inference.py \
  --model best_damage_model.pth \
  --input column/ \
  --output damage_results.csv \
  --detailed

# Input: column/ (main test images)
# Output: damage_results_detailed.csv
```

### Step 7: Run Crack Classification

```bash
# Classify extracted sub-images for crack types
python Single_Head_Classifier/Crack_Single_Head/crack_inference.py \
  --model best_crack_model.pth \
  --input extracted_boxes/ \
  --output crack_results.csv \
  --detailed

# Input: extracted_boxes/ (sub-images from step 3)
# Output: crack_results_detailed.csv
```

### Step 8: Combine Results

```bash
# Merge crack and damage predictions with mapping rules
python Single_Head_Classifier/csv_combiner.py

# Input: crack_results_detailed.csv, damage_results_detailed.csv
# Output: combined_predictions.csv
```

## File Structure

```
Competition_4/
├── Yolo_v8_l_Detection/
│   ├── 1. yolov8_test_script.py        # YOLOv8 detection
│   ├── 2. extract_boxes_script.py      # Extract bounding boxes
│   └── concrete_damage.yaml            # YOLO config
├── Single_Head_Classifier/
│   ├── Damage_Single_Head/
│   │   ├── damage_classifier.py        # Train damage model
│   │   └── damage_inference.py         # Damage inference
│   ├── Crack_Single_Head/
│   │   ├── crack_classifier.py         # Train crack model
│   │   └── crack_inference.py          # Crack inference
│   ├── Balancing_Dataset/
│   │   ├── simple_balancer.py          # Dataset balancing
│   │   └── test.ipynb                  # Interactive testing
│   └── csv_combiner.py                 # Combine results
└── column/                             # Test images
```

## Class Definitions

### Detection Classes (YOLOv8)
- 0: Exposed rebar
- 1: Crack
- 2: Spalling

### Damage Classes (ResNet50)
- 18: Severe Damage
- 19: Moderate Damage
- 20: Minor Damage

### Crack Classes (ResNet50)
- 3: X Shape
- 4: Continuous Diagonal Crack
- 5: Discontinuous Diagonal Crack
- 6: Continuous Vertical Crack
- 7: Discontinuous Vertical Crack
- 8: Continuous Horizontal Crack
- 9: Discontinuous Horizontal Crack

## Key Output Files

| File | Description | Format |
|------|-------------|--------|
| `detection_report.csv` | YOLOv8 detection results | image_id, detected_classes |
| `extracted_boxes_report.csv` | Bounding box mapping | main_image, sub_images, class_ids |
| `damage_results_detailed.csv` | Damage classification | image_name, damage_class, confidence |
| `crack_results_detailed.csv` | Crack classification | image_name, crack_class, confidence |
| `combined_predictions.csv` | Final merged results | all_classes, mapped_classes, high_prob_classes |

## Mapping Rules

The system applies strict rules when combining damage and crack predictions:
- Damage 18 can only have crack classes 3, 4, 5, 6
- Damage 19 can only have crack classes 5, 7
- Damage 20 can only have crack class 9

## Configuration

### YOLOv8 Config (concrete_damage.yaml)
```yaml
path: /path/to/concrete_damage_dataset
train: train/images
val: val/images
test: test/images
nc: 3
names:
  0: Exposed rebar
  1: Crack
  2: Spalling
```

### Model Parameters
- YOLOv8: Image size 1024px, batch size 24
- ResNet50: Image size 224px, batch size 32
- Training: Early stopping, weighted loss for imbalanced data

## Quick Start

1. Put test images in `column/` directory
2. Run detection: `python "Yolo_v8_l_Detection/1. yolov8_test_script.py"`
3. Extract boxes: `python "Yolo_v8_l_Detection/2. extract_boxes_script.py"`
4. Run damage inference: `python Single_Head_Classifier/Damage_Single_Head/damage_inference.py --model best_damage_model.pth --input column/ --output damage_results.csv --detailed`
5. Run crack inference: `python Single_Head_Classifier/Crack_Single_Head/crack_inference.py --model best_crack_model.pth --input extracted_boxes/ --output crack_results.csv --detailed`
6. Combine results: `python Single_Head_Classifier/csv_combiner.py`
7. Check final results in `combined_predictions.csv`

## Troubleshooting

- **CUDA memory error**: Reduce batch size in training scripts
- **Model not found**: Ensure model weights exist in specified paths
- **No images found**: Check input directory paths and file extensions
- **CSV format error**: Verify column names match expected format
- **Low accuracy**: Check dataset balance and quality

## Interactive Testing

Use the Jupyter notebook for experimentation:
```python
# In Single_Head_Classifier/Balancing_Dataset/test.ipynb
from simple_balancer import SimpleDataBalancer

balancer = SimpleDataBalancer("classification", "classification_balanced")
balancer.analyze_distribution()
balancer.balance_equal(target_size=500, augmentation_level="medium")
```
