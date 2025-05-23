import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.models import efficientnet_v2_m
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multi-label mapping for crack types
MULTI_LABEL_MAPPING = {
    0: 4,
    1: 5,
    2: 8,
    3: 9,
    4: 6,
    5: 7,
    6: 10,
    7: 3,

    # IGNORE these classes for now
    # "spalling-like_cracks"
    # "Web_large"
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Stronger data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Slightly stronger test transform (center crop)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def stratified_split(dataset, test_size=0.2, seed=42):
    targets = [label for _, label in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = list(range(len(dataset)))
    train_idx, val_idx = next(sss.split(indices, targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx), targets

def train_multi_label_model(num_epochs=50):
    """Train the multi-label model for crack classification with improved accuracy"""
    try:
        logger.info("Starting multi-label model training (improved)...")

        data_path = './datasets/crack_classification/resized/wall_resize'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset directory not found at {data_path}")

        full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        train_dataset, val_dataset, targets = stratified_split(full_dataset, test_size=0.18, seed=42)

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        logger.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        logger.info(f"Class distribution: {np.unique(targets, return_counts=True)}")
        logger.info(f"Class weights: {class_weights}")

        # Use EfficientNetV2-M for better accuracy
        model = efficientnet_v2_m(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)

        # Enable full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

        model = model.to(device)

        # Use label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=4)

        scaler = torch.cuda.amp.GradScaler()  # For mixed precision

        best_loss = float('inf')
        early_stop_patience = 10
        no_improve_count = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total

            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            # Save model if validation loss improves
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model_multi_label.pth')
                logger.info(f"Model improved and saved as best_model_multi_label.pth")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            scheduler.step(val_loss)

        logger.info("Multi-label model training completed")
        return model

    except Exception as e:
        logger.error(f"Error during multi-label model training: {e}")
        raise

def test_multi_label_model(yolo_results, yolo_cropped_images):
    """
    Process cropped images from YOLO detection and classify them using the multi-label model.
    
    Args:
        yolo_results: Dictionary mapping file_id to list of class IDs
        yolo_cropped_images: Dictionary mapping file_id to list of cropped images
        
    Returns:
        Dictionary mapping file_id to list of updated class IDs
    """
    try:
        # Load model
        model = efficientnet_v2_m()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)

        # Try to load the trained model
        model_path = 'best_model_multi_label.pth'
        if not os.path.exists(model_path):
            train_multi_label_model(num_epochs=50)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("Multi-label model loaded successfully.")

        # Copy results to avoid modifying the original
        multi_label_results = {}

        with torch.no_grad():
            # Process files in sorted order by file_id for consistent results
            for file_id in sorted(yolo_results.keys()):
                class_ids = yolo_results[file_id]
                # Initialize with empty list if not already present
                multi_label_results[file_id] = []
                
                # Check if we have class ID 0 (Exposed Rebar) or 2 (Huge Spalling)
                # If so, we keep those and skip crack classification
                if len(class_ids) == 0:
                    multi_label_results[file_id] = [20, 1]
                    logger.info(f"File ID {file_id}: No cracks found, assigning into label 1: no significant damage, class 20")
                    continue
                elif 0 in class_ids or 2 in class_ids:
                    # Keep exposed rebar and huge spalling, filter out cracks (class 1)
                    for cls in class_ids:
                        if cls != 1:  # If not crack
                            if cls not in multi_label_results[file_id]:  # Only add unique items
                                multi_label_results[file_id].append(cls)
                    
                    # Sort the results
                    multi_label_results[file_id].sort()
                    # add 18 in front
                    multi_label_results[file_id] = [18] + multi_label_results[file_id]

                    logger.info(f"File ID {file_id}: Found crack types {multi_label_results[file_id]}, with exposed rebar or huge spalling class 18")
                    continue

                # If we found cracks (class 1), process with multi-label model
                if 1 in class_ids:
                    cropped_images = yolo_cropped_images.get(file_id, [])
                    crack_classes = []
                    
                    for img in cropped_images:
                        try:
                            # Process each cropped image
                            tensor = test_transform(img).unsqueeze(0).to(device)
                            output = model(tensor)
                            _, pred = torch.max(output, 1)
                            pred_idx = pred.item()
                            
                            # Map the prediction to our multi-label ID
                            crack_classes.append(MULTI_LABEL_MAPPING[pred_idx])
                        except Exception as e:
                            logger.error(f"Error processing crack image: {e}")
                    
                    # Remove duplicates and sort
                    multi_label_results[file_id] = sorted(list(set(crack_classes)))
                    logger.info(f"File ID {file_id}: Found crack types {multi_label_results[file_id]}")

        return multi_label_results

    except Exception as e:
        logger.error(f"Error in multi-label model processing: {e}")
        raise
