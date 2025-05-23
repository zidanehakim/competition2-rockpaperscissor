import os
import logging
import torch
from ultralytics import YOLO
from PIL import Image

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define dataset paths
train_images_path = "./datasets/damage_detection/3class/for_training/datasets_3class_wall_shuffled/train/images/"
train_labels_path = "./datasets/damage_detection/3class/for_training/datasets_3class_wall_shuffled/train/labels/"
test_images_path = "./datasets/test_data/wall/"
data_yaml_path = "./datasets/damage_detection/3class/for_training/datasets_3class_wall_shuffled/data.yaml"
weights_path = "./epoch160.pt"

YOLO_MAPPING = {
    0: "Exposed Rebar",
    1: "Cracks",
    2: "Huge Spalling",
}

def validate_paths():
    paths = [train_images_path, train_labels_path, data_yaml_path]
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            raise FileNotFoundError(f"Path does not exist: {path}")
    logger.info("All dataset paths validated successfully.")

def check_model_weights():
    if os.path.exists(weights_path):
        logger.info(f"Model weights found at {weights_path}.")
        return True
    else:
        logger.warning(f"Model weights not found at {weights_path}. Starting new training.")
        return False

def train_yolo_model():
    """
    Train YOLOv8x model with strong augmentation and best hyperparameters for high mAP.
    """
    try:
        validate_paths()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Use YOLOv11x (large, less memory than x) pretrained on COCO
        model = YOLO("yolo11x.pt")

        # Strong augmentation and robust hyperparameters
        results = model.train(
            data=data_yaml_path,
            epochs=200,
            imgsz=640,                # Reduced image size for memory
            batch=4,                  # Reduced batch size for memory
            device=device,
            name="yolo-crack-best",
            workers=2,                # Reduce if still OOM
            project="runs/train",
            exist_ok=True,
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.949,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            dropout=0.2,
            multi_scale=True,
            augment=True,
            hsv_h=0.1,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=30.0,
            translate=0.3,
            scale=0.8,
            shear=10.0,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.5,
            copy_paste=0.3,
            patience=50,
            val=True,
            save_period=10,
            seed=42,
            verbose=True,
            close_mosaic=10,
        )

        if device == "cuda":
            logger.info(f"Final GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"Final GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Save the trained model
        model.save(weights_path)
        logger.info("Model trained and saved successfully.")

        # Log mAP and mAP50
        try:
            metrics = results.metrics if hasattr(results, "metrics") else None
            if metrics:
                logger.info(f"Final mAP: {metrics['metrics/mAP_0.5-0.95']:.4f}, mAP50: {metrics['metrics/mAP_0.5']:.4f}")
        except Exception as e:
            logger.warning(f"Could not log mAP metrics: {e}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def test_yolo_model():
    """
    Run inference on test images, crop detected cracks, and return results.
    Args:
        test_images_path: Path to test images directory.
    Returns:
        yolo_results: dict mapping file_id to list of detected class IDs.
        yolo_cropped_images: dict mapping file_id to list of cropped PIL images.
    """
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        model = YOLO(weights_path)
        results = model(test_images_path, conf=0.25, iou=0.45)

        yolo_results = {}
        yolo_cropped_images = {}
        for result in results:
            file_path = result.path
            file_name = os.path.basename(file_path)
            file_id = int(os.path.splitext(file_name)[0])
            img = Image.open(file_path).convert("RGB")

            class_ids = []
            crops = []
            
            # Create output directory for visualizations
            output_dir = os.path.join("./outputs", "predictions")
            os.makedirs(output_dir, exist_ok=True)

            # Save the annotated image with detection boxes
            annotated_img = result.plot()  # This returns the image with detection boxes drawn
            annotated_pil = Image.fromarray(annotated_img)
            annotated_pil.save(os.path.join(output_dir, f"{file_id}_annotated.jpg"))

            if result.boxes is not None and len(result.boxes) > 0:
                for i, (box, cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int))):
                    x1, y1, x2, y2 = map(int, box)
                    cropped = img.crop((x1, y1, x2, y2))
                    crops.append(cropped)
                    class_ids.append(int(cls))

            yolo_results[file_id] = class_ids
            yolo_cropped_images[file_id] = crops

            logger.info(f"{file_name} (ID: {file_id}): {len(class_ids)} detections. Saved to {output_dir}")

        logger.info(f"All predictions saved to ./outputs/predictions")
        return yolo_results, yolo_cropped_images

    except Exception as e:
        logger.error(f"Error during YOLO inference: {e}")
        raise