#!/usr/bin/env python3
"""
Compact YOLOv8 Test Script - Sequenced CSV with All Values
"""

import os
import csv
import re
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def numeric_sort(filename):
    """Extract number from filename for sorting"""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def test_yolov8(model_path="runs_1/detect/train/weights/best.pt", 
                input_folder="column", 
                output_csv="results.csv",
                conf_threshold=0.25):
    """Test YOLOv8 and generate sequenced CSV with all detection values"""
    
    print(f"🚀 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get sorted image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_folder).glob(f"*{ext}"))
        image_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files, key=lambda x: numeric_sort(x.name))
    print(f"📸 Found {len(image_files)} images")
    
    # Process images and create CSV data
    csv_data = []
    for i, img_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {img_path.name}")
        
        try:
            # Run inference
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            result = results[0]
            
            # Extract detections
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'class_id': int(cls_id),
                        'class_name': model.names[cls_id],
                        'confidence': float(conf),
                        'box': [float(x) for x in box]
                    })
            
            # Prepare CSV row
            class_ids = [str(d['class_id']) for d in detections]
            class_names = [d['class_name'] for d in detections]
            confidences = [f"{d['confidence']:.3f}" for d in detections]
            boxes = [f"({d['box'][0]:.0f},{d['box'][1]:.0f},{d['box'][2]:.0f},{d['box'][3]:.0f})" for d in detections]
            
            csv_row = {
                'ID': i,
                'Image_Name': img_path.name,
                'Detection_Count': len(detections),
                'Class_IDs': ','.join(class_ids),
                'Class_Names': ','.join(class_names),
                'Confidences': ','.join(confidences),
                'Bounding_Boxes': ','.join(boxes),
                'Avg_Confidence': f"{np.mean([d['confidence'] for d in detections]):.3f}" if detections else "0.000",
                'Max_Confidence': f"{max([d['confidence'] for d in detections]):.3f}" if detections else "0.000"
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            csv_row = {
                'ID': i, 'Image_Name': img_path.name, 'Detection_Count': 0,
                'Class_IDs': '', 'Class_Names': '', 'Confidences': '', 
                'Bounding_Boxes': '', 'Avg_Confidence': '0.000', 'Max_Confidence': '0.000'
            }
        
        csv_data.append(csv_row)
    
    # Save CSV
    headers = ['ID', 'Image_Name', 'Detection_Count', 'Class_IDs', 'Class_Names', 
               'Confidences', 'Bounding_Boxes', 'Avg_Confidence', 'Max_Confidence']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Print summary
    total_detections = sum(row['Detection_Count'] for row in csv_data)
    images_with_detections = sum(1 for row in csv_data if row['Detection_Count'] > 0)
    
    print(f"\n✅ Results saved to: {output_csv}")
    print(f"📊 Summary: {len(csv_data)} images, {total_detections} detections")
    print(f"   Images with detections: {images_with_detections}/{len(csv_data)}")
    
    # Show first 3 results
    print(f"\n📋 Sample results:")
    for row in csv_data[:3]:
        print(f"   {row['ID']}. {row['Image_Name']}: {row['Detection_Count']} detections")
        if row['Class_IDs']:
            print(f"      Classes: {row['Class_IDs']} | Conf: {row['Confidences']}")
    
    return csv_data

def analyze_single(model_path, image_path, output_csv="single_result.csv"):
    """Analyze single image"""
    return test_yolov8(model_path, Path(image_path).parent, output_csv)

if __name__ == "__main__":
    # Quick configuration
    CONFIG = {
        "model_path": "runs/detect/train/weights/best.pt",
        "input_folder": "column",
        "output_csv": "detection_results.csv",
        "conf_threshold": 0.25
    }
    
    print("🚀 YOLOv8 Testing with Sequenced CSV Output")
    csv_data = test_yolov8(**CONFIG)
    print("🎉 Complete!")

# Usage examples:
# python compact_yolov8_test.py
# test_yolov8("model.pt", "images/", "results.csv", 0.3)
# analyze_single("model.pt", "test.jpg")
