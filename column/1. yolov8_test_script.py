import os
import json
import csv
import cv2
from datetime import datetime
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def test_yolov8_concrete_damage(
    model_path="runs/detect/train/weights/best.pt",  # or "runs/detect/train/weights/last.pt"
    input_folder="column",
    output_json_folder="results_json",
    output_images_folder="results_images",
    confidence_threshold=0.25,
    iou_threshold=0.45
):
    """
    Test YOLOv8 model on concrete damage detection dataset
    
    Args:
        model_path (str): Path to the YOLOv8 model weights (.pt file)
        input_folder (str): Folder containing test images
        output_json_folder (str): Folder to save JSON results
        output_images_folder (str): Folder to save annotated images
        confidence_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
    """
    
    # Create output directories
    os.makedirs(output_json_folder, exist_ok=True)
    os.makedirs(output_images_folder, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    
    # Get class names from the model
    class_names = model.names
    print(f"Model classes: {class_names}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files from input folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f"*{ext}"))
        image_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # Run inference
            results = model(
                str(image_path),
                conf=confidence_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False
            )
            
            # Process results for current image
            image_results = process_single_image_results(
                results[0], 
                image_path, 
                class_names,
                output_images_folder
            )
            
            # Save individual JSON file for this image
            json_filename = f"{image_path.stem}_results.json"
            json_path = os.path.join(output_json_folder, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(image_results, f, indent=2)
            
            all_results.append(image_results)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    # Save combined results
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "total_images": len(image_files),
        "processed_images": len(all_results),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "class_names": class_names,
        "results": all_results
    }
    
    combined_json_path = os.path.join(output_json_folder, "combined_results.json")
    with open(combined_json_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Generate CSV report
    csv_report_path = generate_csv_report(all_results, output_json_folder)
    
    print(f"\nTesting completed!")
    print(f"Individual JSON files saved to: {output_json_folder}")
    print(f"Combined results saved to: {combined_json_path}")
    print(f"CSV report saved to: {csv_report_path}")
    print(f"Annotated images saved to: {output_images_folder}")
    
    # Print summary statistics
    print_summary_statistics(all_results, class_names)

def process_single_image_results(result, image_path, class_names, output_images_folder):
    """
    Process results for a single image and save annotated image
    
    Args:
        result: YOLO result object for single image
        image_path: Path to the original image
        class_names: Dictionary of class names
        output_images_folder: Folder to save annotated images
        
    Returns:
        dict: Processed results for the image
    """
    
    # Read original image
    image = cv2.imread(str(image_path))
    original_height, original_width = image.shape[:2]
    
    # Initialize image results
    image_results = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "image_size": {
            "width": original_width,
            "height": original_height
        },
        "detections": []
    }
    
    # Process detections
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for j, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            # Create detection dictionary
            detection = {
                "detection_id": j + 1,
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "confidence": float(conf),
                "bounding_box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2)
                }
            }
            
            image_results["detections"].append(detection)
            
            # Draw bounding box on image
            color = get_class_color(class_id)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label
            label = f"{class_names[class_id]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(image, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save annotated image
    output_image_path = os.path.join(output_images_folder, f"annotated_{image_path.name}")
    cv2.imwrite(output_image_path, image)
    
    # Add detection count to results
    image_results["detection_count"] = len(image_results["detections"])
    image_results["output_image_path"] = output_image_path
    
    return image_results

def get_class_color(class_id):
    """
    Get consistent color for each class
    
    Args:
        class_id (int): Class ID
        
    Returns:
        tuple: BGR color tuple
    """
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 0),    # Dark Green
        (128, 128, 128) # Gray
    ]
    return colors[class_id % len(colors)]

def generate_csv_report(all_results, output_folder):
    """
    Generate CSV report with image ID and detected classes
    
    Args:
        all_results (list): List of all image results
        output_folder (str): Output folder to save CSV
        
    Returns:
        str: Path to the generated CSV file
    """
    
    csv_path = os.path.join(output_folder, "detection_report.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['image id', 'Classes'])
        
        # Process each image result
        for result in all_results:
            image_name = result["image_name"]
            
            # Extract class IDs from detections
            class_ids = []
            for detection in result["detections"]:
                class_ids.append(str(detection["class_id"]))
            
            # Create classes string (comma-separated class IDs)
            if class_ids:
                classes_str = ",".join(class_ids)
            else:
                classes_str = ""  # No detections
            
            # Write row
            writer.writerow([image_name, classes_str])
    
    print(f"CSV report generated with {len(all_results)} images")
    return csv_path

def print_summary_statistics(all_results, class_names):
    """
    Print summary statistics of the testing results
    
    Args:
        all_results (list): List of all image results
        class_names (dict): Dictionary of class names
    """
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    total_detections = sum(len(result["detections"]) for result in all_results)
    images_with_detections = sum(1 for result in all_results if result["detection_count"] > 0)
    
    print(f"Total images processed: {len(all_results)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Images without detections: {len(all_results) - images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(all_results):.2f}")
    
    # Class-wise statistics
    class_counts = {}
    confidence_sums = {}
    
    for result in all_results:
        for detection in result["detections"]:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                confidence_sums[class_name] = 0.0
            
            class_counts[class_name] += 1
            confidence_sums[class_name] += confidence
    
    if class_counts:
        print(f"\nClass-wise Detection Statistics:")
        print("-" * 40)
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            avg_conf = confidence_sums[class_name] / count
            print(f"{class_name}: {count} detections (avg conf: {avg_conf:.3f})")

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "model_path": "runs/detect/train/weights/best.pt",  # Change to "runs/detect/train/weights/last.pt" if you want to use that model
        "input_folder": "column",
        "output_json_folder": "results_json",
        "output_images_folder": "results_images", 
        "confidence_threshold": 0.25,  # Adjust as needed
        "iou_threshold": 0.45  # Adjust as needed
    }
    
    print("Starting YOLOv8 Concrete Damage Detection Testing...")
    print(f"Configuration: {CONFIG}")
    
    # Run the testing
    test_yolov8_concrete_damage(**CONFIG)
    
    print("\nTesting completed successfully!")

# Additional utility function to analyze specific image
def analyze_single_image(model_path, image_path, output_folder="single_analysis"):
    """
    Analyze a single image and return detailed results
    
    Args:
        model_path (str): Path to model weights
        image_path (str): Path to single image
        output_folder (str): Output folder for results
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    model = YOLO(model_path)
    results = model(image_path, save=False, verbose=True)
    
    result = results[0]
    image_path = Path(image_path)
    
    # Process and save results
    image_results = process_single_image_results(
        result, image_path, model.names, output_folder
    )
    
    # Save JSON
    json_path = os.path.join(output_folder, f"{image_path.stem}_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(image_results, f, indent=2)
    
    print(f"Single image analysis completed!")
    print(f"Results saved to: {json_path}")
    
    return image_results