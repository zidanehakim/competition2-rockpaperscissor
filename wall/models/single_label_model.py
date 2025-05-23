import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.models import efficientnet_v2_m, efficientnet_v2_s
from PIL import Image
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label restrictions
LABEL_RESTRICTIONS = {
    18: [0,2],  # Class A
    19: [3,4,6,8],  # Class B
    20: [1,5,7,9,10],  # Class C
}

# Single label mapping for damage classes
SINGLE_LABEL_MAPPING = {
    "Class A": 18,
    "Class B": 19,
    "Class C": 20,
}

# Inverse mapping for easier reference
IDX_TO_SINGLE_LABEL = {v: k for k, v in SINGLE_LABEL_MAPPING.items()}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
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
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transform for testing/validation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_single_label_model(num_epochs=40):
    """Train the single-label model for damage classification with improved accuracy"""
    try:
        logger.info("Starting single-label model training...")

        data_path = './datasets/damage_classification_forTrain/wall_damage'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset directory not found at {data_path}")

        # Load dataset for stratified split
        full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        targets = [label for _, label in full_dataset.samples]

        # Stratified split for balanced validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = next(sss.split(indices, targets))
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(datasets.ImageFolder(data_path, transform=test_transform), val_idx)

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        logger.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        logger.info(f"Class distribution: {np.unique(targets, return_counts=True)}")
        logger.info(f"Class weights: {class_weights}")

        # Use EfficientNetV2-M if available, else fallback to S
        try:
            model = efficientnet_v2_m(weights='IMAGENET1K_V1')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        except Exception:
            model = efficientnet_v2_s(weights='IMAGENET1K_V1')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

        # Unfreeze last 2 blocks + classifier for better feature learning
        for name, param in model.named_parameters():
            if "classifier" in name or "features.6" in name or "features.5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model = model.to(device)

        # Use label smoothing
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        best_loss = float('inf')
        best_acc = 0.0
        early_stop_patience = 8
        no_improve_count = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Mixup augmentation
                if random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                # For mixup, use original labels for accuracy
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
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
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

            # Save best model by validation accuracy
            if val_acc > best_acc or val_loss < best_loss:
                best_loss = min(val_loss, best_loss)
                best_acc = max(val_acc, best_acc)
                torch.save(model.state_dict(), 'best_model_single_label.pth')
                logger.info(f"Model improved and saved as best_model_single_label.pth (Val Acc: {val_acc:.2f}%)")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            scheduler.step()

        logger.info(f"Single-label model training completed. Best Val Acc: {best_acc:.2f}%")
        return model

    except Exception as e:
        logger.error(f"Error during single-label model training: {e}")
        raise

def test_single_label_model(multi_label_results):
    """
    Process the images using the single-label model and combine with multi-label results.
    
    Args:
        multi_label_results: Dictionary mapping file_id to list of class IDs from multi-label model
        
    Returns:
        Dictionary mapping file_id to formatted class string for final submission
    """
    try:
        # Try to load the trained model with the same architecture as in training
        model_path = 'best_model_single_label.pth'
        model = None
        if not os.path.exists(model_path):
            train_single_label_model(num_epochs=20)
        try:
            model = efficientnet_v2_m()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            model = efficientnet_v2_s()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
            model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("Single-label model loaded successfully.")

        # Path to test images directory
        test_data_dir = './datasets/test_data/wall'
        
        # Process images and update multi-label results
        with torch.no_grad():
            for file_id in list(multi_label_results.keys()):
                try:
                    # If already processed by single-label, skip
                    if 18 in multi_label_results[file_id] or 19 in multi_label_results[file_id] or 20 in multi_label_results[file_id]:
                        continue
                    # Load and process the full image
                    image_path = os.path.join(test_data_dir, f"{file_id}.jpg")
                    image = Image.open(image_path).convert('RGB')
                    tensor = test_transform(image).unsqueeze(0).to(device)

                    # Get prediction from single-label model
                    output = model(tensor)
                    _, pred = torch.max(output, 1)
                    pred_idx = pred.item()
                    damage_class = pred_idx + 18  # Map to 18/19/20

                    # Check if all multi-labels are in the same class
                    # if all(label in LABEL_RESTRICTIONS[18] for label in multi_label_results[file_id]):
                    #     damage_class = 18
                    # elif all(label in LABEL_RESTRICTIONS[19] for label in multi_label_results[file_id]):
                    #     damage_class = 19
                    # elif all(label in LABEL_RESTRICTIONS[20] for label in multi_label_results[file_id]):
                    #     damage_class = 20
                    # If mixed, use single-label prediction as main class

                    # Filter multi-labels to keep only those in the predicted class
                    filtered_labels = [label for label in multi_label_results[file_id] if label in LABEL_RESTRICTIONS[damage_class]]

                    # Ensure no duplicates and main class is first
                    updated_labels = [damage_class] + [label for label in filtered_labels if label != damage_class]

                    multi_label_results[file_id] = updated_labels

                    # If only [18], make it [18,0], if only [20], make it [20,10]
                    # This is because we trust the single-label model more, so we add default labels rather than having wrong 100% wrong answer
                    if len(multi_label_results[file_id]) == 1:
                        if multi_label_results[file_id][0] == 18:
                            multi_label_results[file_id].append(0)
                        elif multi_label_results[file_id][0] == 20:
                            multi_label_results[file_id].append(10)

                    logger.info(f"File: {file_id}.jpg - Predicted damage class: {IDX_TO_SINGLE_LABEL.get(damage_class, 'Unknown')}, Multi-label classes: {multi_label_results[file_id]}")
                except Exception as e:
                    logger.error(f"Error processing image {file_id}.jpg: {e}")
                    continue
        
        return multi_label_results
        
    except Exception as e:
        logger.error(f"Error in single-label model processing: {e}")
        raise
