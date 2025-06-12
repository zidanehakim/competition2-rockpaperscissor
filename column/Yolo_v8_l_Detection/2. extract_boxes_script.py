import os
import json
import csv
import cv2
from pathlib import Path
import numpy as np

def resize_with_padding(image, target_size=(640, 640), pad_color=(0, 0, 0)):
    """
    Resize image while maintaining aspect ratio and add padding to reach target size
    
    Args:
        image: Input image (numpy array)
        target_size: Target size as (width, height)
        pad_color: Padding color as (B, G, R) for black padding
        
    Returns:
        Resized image with padding to exact target size
    """
    
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]
    
    # Calculate scaling factor to fit image within target size
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)  # Use smaller scale to fit within bounds
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image with calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create target size image filled with padding color
    padded_image = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    
    # Calculate position to center the resized image
    start_x = (target_width - new_width) // 2
    start_y = (target_height - new_height) // 2
    
    # Place resized image in center of padded image
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image
    
    return padded_image

def extract_bounding_boxes_from_json(
    json_file_path="results_json/combined_results.json",
    original_images_folder="column", 
    output_folder="extracted_boxes",
    output_csv="extracted_boxes_report.csv",
    target_size=(640, 640)
):
    """
    Extract bounding box regions from images using combined JSON detection results
    
    Args:
        json_file_path (str): Path to the combined JSON results file
        original_images_folder (str): Folder containing original images
        output_folder (str): Folder to save extracted bounding box images
        output_csv (str): Path for the output CSV file
        target_size (tuple): Target size for resized images (width, height)
    """
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load JSON file
    print(f"Loading JSON file: {json_file_path}")
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return
    
    # Extract results from JSON
    if "results" in data:
        results = data["results"]
    else:
        print("Error: 'results' key not found in JSON file")
        return
    
    print(f"Found {len(results)} images in JSON file")
    
    # Process each image
    csv_data = []
    extracted_count = 0
    
    for image_result in results:
        image_name = image_result["image_name"]
        detections = image_result.get("detections", [])
        
        print(f"Processing: {image_name} ({len(detections)} detections)")
        
        # Load original image
        image_path = os.path.join(original_images_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Original image not found: {image_path}")
            continue
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue
        
        # Prepare CSV row data
        image_id = Path(image_name).stem  # Remove extension
        
        # Get all class IDs for Detection column
        all_class_ids = [str(detection["class_id"]) for detection in detections]
        detection_string = ",".join(all_class_ids) if all_class_ids else ""
        
        csv_row = {
            "Input Id": image_name,
            "Detection": detection_string
        }
        
        # Extract each bounding box
        for i, detection in enumerate(detections):
            detection_id = i + 1
            bbox = detection["bounding_box"]
            class_name = detection["class_name"]
            
            # Extract bounding box coordinates
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"]) 
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            
            # Ensure coordinates are within image bounds
            h, w = original_image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Extract bounding box region
            bbox_image = original_image[y1:y2, x1:x2]
            
            if bbox_image.size == 0:
                print(f"Warning: Empty bounding box for {image_name}, detection {detection_id}")
                continue
            
            # Resize to target size with aspect ratio preservation
            resized_bbox = resize_with_padding(bbox_image, target_size)
            
            # Generate output filename
            output_filename = f"{image_id}_{detection_id}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save extracted and resized bounding box
            cv2.imwrite(output_path, resized_bbox)
            
            # Add to CSV row
            csv_row[f"id_{detection_id}"] = output_filename
            csv_row[f"detect_{detection_id}"] = detection["class_id"]  # Use class_id instead of class_name
            
            extracted_count += 1
            
        csv_data.append(csv_row)
    
    # Generate CSV file
    generate_csv_report(csv_data, output_csv)
    
    print(f"\n✅ Extraction completed!")
    print(f"📁 Extracted {extracted_count} bounding boxes to: {output_folder}")
    print(f"📄 CSV report saved to: {output_csv}")
    print(f"🔧 All images resized to: {target_size[0]}x{target_size[1]} (aspect ratio preserved with black padding)")
    print(f"📋 CSV format: Input Id, Detection (class IDs), id_X, detect_X (class ID)")

def generate_csv_report(csv_data, output_csv):
    """
    Generate CSV report with dynamic columns based on maximum detections
    
    Args:
        csv_data (list): List of dictionaries containing CSV row data
        output_csv (str): Path for output CSV file
    """
    
    if not csv_data:
        print("No data to write to CSV")
        return
    
    # Find maximum number of detections to determine columns
    max_detections = max(len(row["Detection"].split(",")) if row["Detection"] else 0 for row in csv_data)
    
    # Build column headers
    headers = ["Input Id", "Detection"]
    for i in range(1, max_detections + 1):
        headers.extend([f"id_{i}", f"detect_{i}"])
    
    # Write CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for row in csv_data:
            # Fill missing columns with empty strings
            complete_row = {header: row.get(header, "") for header in headers}
            writer.writerow(complete_row)
    
    print(f"CSV report generated with {len(csv_data)} rows and {len(headers)} columns")

def extract_single_image_boxes(
    image_path,
    detections_data,
    output_folder="single_extraction",
    target_size=(640, 640)
):
    """
    Extract bounding boxes from a single image
    
    Args:
        image_path (str): Path to the image
        detections_data (list): List of detection dictionaries
        output_folder (str): Output folder for extracted boxes
        target_size (tuple): Target size for resized images
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image_name = Path(image_path).stem
    extracted_files = []
    
    for i, detection in enumerate(detections_data):
        detection_id = i + 1
        bbox = detection["bounding_box"]
        
        # Extract coordinates
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        
        # Extract and resize with padding
        bbox_image = image[y1:y2, x1:x2]
        resized_bbox = resize_with_padding(bbox_image, target_size)
        
        # Save
        output_filename = f"{image_name}_{detection_id}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, resized_bbox)
        
        extracted_files.append(output_filename)
        print(f"Extracted: {output_filename}")
    
    return extracted_files

def preview_json_structure(json_file_path):
    """
    Preview the structure of the JSON file to understand the data format
    
    Args:
        json_file_path (str): Path to the JSON file
    """
    
    print("🔍 PREVIEWING JSON STRUCTURE")
    print("=" * 50)
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Top-level keys: {list(data.keys())}")
        
        if "results" in data:
            results = data["results"]
            print(f"Number of images: {len(results)}")
            
            if results:
                sample_result = results[0]
                print(f"\nSample image structure:")
                print(f"  Image name: {sample_result.get('image_name', 'N/A')}")
                print(f"  Detections count: {len(sample_result.get('detections', []))}")
                
                if sample_result.get('detections'):
                    sample_detection = sample_result['detections'][0]
                    print(f"  Sample detection keys: {list(sample_detection.keys())}")
                    print(f"  Sample bounding box keys: {list(sample_detection.get('bounding_box', {}).keys())}")
                    print(f"  Sample class_id: {sample_detection.get('class_id', 'N/A')}")
                    print(f"  Sample class_name: {sample_detection.get('class_name', 'N/A')}")
        
        print("\n✅ JSON structure looks good!")
        
    except Exception as e:
        print(f"❌ Error reading JSON: {str(e)}")

# Configuration and main execution
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "json_file_path": "results_json/combined_results.json",
        "original_images_folder": "column",
        "output_folder": "extracted_boxes",
        "output_csv": "extracted_boxes_report.csv",
        "target_size": (640, 640)  # Width x Height
    }
    
    print("🔧 BOUNDING BOX EXTRACTION TOOL (ASPECT RATIO PRESERVED)")
    print("=" * 65)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("\n🎯 Feature: Maintains aspect ratio with black padding")
    print("   Long cracks → black bars on sides")
    print("   Wide cracks → black bars on top/bottom")
    print("   Square damage → no padding needed\n")
    
    # Optional: Preview JSON structure first
    # preview_json_structure(CONFIG["json_file_path"])
    
    # Run extraction
    extract_bounding_boxes_from_json(**CONFIG)
    
    print("\n🎉 Process completed successfully!")
    print("📋 All extracted images maintain their original proportions!")

# Example usage for single image
def example_single_image_usage():
    """
    Example of how to extract boxes from a single image
    """
    
    # Example detection data structure
    sample_detections = [
        {
            "class_name": "crack",
            "class_id": 0,
            "confidence": 0.89,
            "bounding_box": {"x1": 100, "y1": 150, "x2": 300, "y2": 250}
        },
        {
            "class_name": "spalling",
            "class_id": 1,
            "confidence": 0.76,
            "bounding_box": {"x1": 400, "y1": 200, "x2": 600, "y2": 350}
        }
    ]
    
    extract_single_image_boxes(
        image_path="column/sample.jpg",
        detections_data=sample_detections,
        output_folder="single_test"
    )
