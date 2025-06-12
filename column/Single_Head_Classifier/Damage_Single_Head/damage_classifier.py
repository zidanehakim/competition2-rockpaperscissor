import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DamageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Damage class mappings
        self.class_to_idx = {'18': 0, '19': 1, '20': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Convert labels to indices
        self.label_indices = [self.class_to_idx[str(label)] for label in labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.label_indices[idx], dtype=torch.long)
        
        return image, label

class SingleHeadResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SingleHeadResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Get classification output
        output = self.classifier(features)
        return output

def get_transforms():
    """Get transforms for training and validation"""
    
    # Training transforms with heavy augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_damage_dataset(root_dir="classification"):
    """Load dataset for damage classification (18, 19, 20)"""
    
    image_paths = []
    labels = []
    
    print("Loading damage dataset...")
    
    # Valid damage classes
    valid_classes = ['18', '19', '20']
    
    # Get all folders
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    for folder in folders:
        if folder not in valid_classes:
            continue
            
        folder_path = os.path.join(root_dir, folder)
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                glob.glob(os.path.join(folder_path, "*.png"))
        
        for img_path in images:
            image_paths.append(img_path)
            labels.append(folder)
    
    print(f"Loaded {len(image_paths)} images for damage classification")
    return image_paths, labels

def calculate_class_weights(labels, num_classes):
    """Calculate class weights for imbalanced dataset"""
    label_counts = Counter(labels)
    weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
    return torch.FloatTensor(weights)

def create_weighted_sampler(labels):
    """Create weighted sampler for balanced training"""
    label_counts = Counter(labels)
    sample_weights = []
    
    for label in labels:
        weight = 1.0 / label_counts[label]
        sample_weights.append(weight)
    
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

class DamageTrainer:
    def __init__(self, model, train_loader, val_loader, device, class_weights, lr=0.0001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Weighted loss function for class imbalance
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # History tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += images.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total_samples:.3f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_samples += images.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting damage classification training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_damage_model.pth')
                print("✓ Saved best damage model")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 10:
                print("Early stopping triggered")
                break
        
        print("Damage classification training completed!")

def plot_training_history(history, model_name="damage"):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{model_name.title()} Classification - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_title(f'{model_name.title()} Classification - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Load dataset
    image_paths, labels = load_damage_dataset()
    
    # Print class distribution
    print("\nDamage Class Distribution:")
    print(Counter(labels))
    
    # Calculate class weights
    label_indices = [{'18': 0, '19': 1, '20': 2}[str(label)] for label in labels]
    class_weights = calculate_class_weights(label_indices, 3)
    
    print(f"\nDamage class weights: {class_weights}")
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"Train: {len(X_train)} images")
    print(f"Val: {len(X_val)} images") 
    print(f"Test: {len(X_test)} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = DamageDataset(X_train, y_train, train_transform)
    val_dataset = DamageDataset(X_val, y_val, val_transform)
    
    # Create weighted sampler
    sampler = create_weighted_sampler(y_train)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    model = SingleHeadResNet(num_classes=3)
    
    # Create trainer
    trainer = DamageTrainer(
        model, train_loader, val_loader, device, class_weights, LEARNING_RATE
    )
    
    # Train model
    trainer.train(NUM_EPOCHS)
    
    # Plot training history
    plot_training_history(trainer.history, "damage")
    
    # Save training history
    with open('damage_training_history.json', 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    print("\nDamage classification training completed!")
    print("Best model saved as: best_damage_model.pth")
    print("Training history saved as: damage_training_history.png and damage_training_history.json")

if __name__ == "__main__":
    main()

#python damage_classifier.py