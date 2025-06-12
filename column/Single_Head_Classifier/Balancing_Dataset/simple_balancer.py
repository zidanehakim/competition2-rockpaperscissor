#!/usr/bin/env python3
"""
Simple Data Balancer - Easy to use version
Quick balancing with predefined configurations
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import json

class SimpleDataBalancer:
    def __init__(self, source_dir: str, output_dir: str):
        """
        Simple data balancer initialization
        
        Args:
            source_dir: Source directory with class folders
            output_dir: Output directory for balanced dataset
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        print(f"🔧 Simple Data Balancer")
        print(f"   Source: {self.source_dir}")
        print(f"   Output: {self.output_dir}")
    
    def analyze_distribution(self) -> Dict[str, int]:
        """Analyze current class distribution"""
        distribution = {}
        
        for class_folder in self.source_dir.iterdir():
            if not class_folder.is_dir():
                continue
            
            # Count images
            images = []
            for ext in self.image_extensions:
                images.extend(list(class_folder.glob(f"*{ext}")))
                images.extend(list(class_folder.glob(f"*{ext.upper()}")))
            
            distribution[class_folder.name] = len(images)
        
        # Print analysis
        print(f"\n📊 Current Distribution:")
        total = sum(distribution.values())
        for class_name, count in sorted(distribution.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"   Class {class_name}: {count:,} images ({percentage:.1f}%)")
        
        print(f"   Total: {total:,} images")
        if distribution:
            imbalance = max(distribution.values()) / min(distribution.values())
            print(f"   Imbalance ratio: {imbalance:.1f}:1")
        
        return distribution
    
    def balance_to_target(self, targets: Dict[str, int], augmentation_level: str = "medium"):
        """
        Balance dataset to specific target counts
        
        Args:
            targets: Dictionary with target count per class
            augmentation_level: 'light', 'medium', or 'heavy'
        """
        print(f"\n🚀 Balancing to target distribution...")
        print(f"   Augmentation level: {augmentation_level}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        current_dist = self.analyze_distribution()
        
        for class_name, target_count in targets.items():
            if class_name not in current_dist:
                print(f"❌ Class {class_name} not found in source!")
                continue
            
            current_count = current_dist[class_name]
            print(f"\n📁 Processing {class_name}: {current_count} → {target_count}")
            
            # Create class output directory
            class_output_dir = self.output_dir / class_name
            class_output_dir.mkdir(exist_ok=True)
            
            # Get source images
            source_images = self._get_class_images(class_name)
            
            if target_count <= current_count:
                # Undersample
                self._undersample_class(source_images, class_output_dir, target_count, class_name)
            else:
                # Oversample with augmentation
                self._oversample_class(source_images, class_output_dir, target_count, class_name, augmentation_level)
        
        print(f"\n✅ Balancing completed! Output: {self.output_dir}")
    
    def balance_equal(self, target_size: int = None, augmentation_level: str = "medium"):
        """
        Balance all classes to equal size
        
        Args:
            target_size: Target size for all classes (if None, uses current maximum)
            augmentation_level: 'light', 'medium', or 'heavy'
        """
        distribution = self.analyze_distribution()
        
        if target_size is None:
            target_size = max(distribution.values())
        
        print(f"\n🎯 Balancing all classes to {target_size} images each")
        
        # Create targets dictionary
        targets = {class_name: target_size for class_name in distribution.keys()}
        
        self.balance_to_target(targets, augmentation_level)
    
    def balance_smart(self, multiplier: float = 1.5, augmentation_level: str = "medium"):
        """
        Smart balancing based on statistical analysis
        
        Args:
            multiplier: Multiplier for mean (target = mean * multiplier)
            augmentation_level: 'light', 'medium', or 'heavy'
        """
        distribution = self.analyze_distribution()
        counts = list(distribution.values())
        
        # Calculate smart target (mean * multiplier, but at least min + some buffer)
        mean_count = np.mean(counts)
        min_count = min(counts)
        smart_target = max(int(mean_count * multiplier), min_count + 100)
        
        print(f"\n🧠 Smart balancing to {smart_target} images per class")
        print(f"   Based on: mean({mean_count:.0f}) × {multiplier} = {smart_target}")
        
        # Create targets dictionary
        targets = {class_name: smart_target for class_name in distribution.keys()}
        
        self.balance_to_target(targets, augmentation_level)
    
    def _get_class_images(self, class_name: str) -> List[Path]:
        """Get all images for a class"""
        class_dir = self.source_dir / class_name
        images = []
        
        for ext in self.image_extensions:
            images.extend(list(class_dir.glob(f"*{ext}")))
            images.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        return images
    
    def _undersample_class(self, source_images: List[Path], output_dir: Path, 
                          target_count: int, class_name: str):
        """Undersample a class by random selection"""
        selected = random.sample(source_images, target_count)
        
        for i, img_path in enumerate(tqdm(selected, desc=f"Copying {class_name}")):
            output_path = output_dir / f"{class_name}_{i+1:04d}{img_path.suffix}"
            shutil.copy2(img_path, output_path)
        
        print(f"   ⬇️  Undersampled: {len(source_images)} → {target_count}")
    
    def _oversample_class(self, source_images: List[Path], output_dir: Path,
                         target_count: int, class_name: str, aug_level: str):
        """Oversample a class using augmentation"""
        current_count = len(source_images)
        
        # Copy all originals first
        for i, img_path in enumerate(source_images):
            output_path = output_dir / f"{class_name}_orig_{i+1:04d}{img_path.suffix}"
            shutil.copy2(img_path, output_path)
        
        # Calculate augmentations needed
        augs_needed = target_count - current_count
        
        if augs_needed <= 0:
            return
        
        print(f"   ⬆️  Need {augs_needed} augmented images")
        
        # Generate augmentations
        aug_count = 0
        pbar = tqdm(total=augs_needed, desc=f"Augmenting {class_name}")
        
        while aug_count < augs_needed:
            for img_path in source_images:
                if aug_count >= augs_needed:
                    break
                
                # Generate one augmented image
                try:
                    augmented = self._augment_image(img_path, aug_level)
                    output_path = output_dir / f"{class_name}_aug_{aug_count+1:04d}.jpg"
                    augmented.save(output_path, 'JPEG', quality=95)
                    aug_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"❌ Error augmenting {img_path}: {e}")
        
        pbar.close()
        print(f"   ✅ Generated {aug_count} augmented images")
    
    def _augment_image(self, image_path: Path, aug_level: str) -> Image.Image:
        """Apply random augmentation to an image"""
        image = Image.open(image_path).convert('RGB')
        
        # Define augmentation options by level
        light_augs = [self._rotate, self._brightness, self._contrast]
        medium_augs = light_augs + [self._flip_h, self._blur, self._saturation]
        heavy_augs = medium_augs + [self._flip_v, self._noise, self._crop_resize]
        
        # Select augmentations based on level
        if aug_level == "light":
            augs = light_augs
            num_augs = random.randint(1, 2)
        elif aug_level == "medium":
            augs = medium_augs
            num_augs = random.randint(1, 3)
        else:  # heavy
            augs = heavy_augs
            num_augs = random.randint(2, 4)
        
        # Apply random augmentations
        for _ in range(num_augs):
            aug_func = random.choice(augs)
            image = aug_func(image)
        
        return image
    
    # Simple augmentation functions
    def _rotate(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-25, 25)
        return img.rotate(angle, fillcolor=(255, 255, 255))
    
    def _flip_h(self, img: Image.Image) -> Image.Image:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def _flip_v(self, img: Image.Image) -> Image.Image:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    def _brightness(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def _contrast(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def _saturation(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Color(img).enhance(factor)
    
    def _blur(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _noise(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        noise = np.random.normal(0, 10, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def _crop_resize(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        crop_factor = random.uniform(0.85, 0.95)
        new_w, new_h = int(w * crop_factor), int(h * crop_factor)
        
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)

# Predefined configuration templates
BALANCE_CONFIGS = {
    "equal_max": {
        "description": "Balance all classes to match the largest class",
        "method": "balance_equal",
        "params": {"target_size": None, "augmentation_level": "medium"}
    },
    
    "equal_1000": {
        "description": "Balance all classes to 1000 images each",
        "method": "balance_equal", 
        "params": {"target_size": 1000, "augmentation_level": "medium"}
    },
    
    "equal_2000": {
        "description": "Balance all classes to 2000 images each",
        "method": "balance_equal",
        "params": {"target_size": 2000, "augmentation_level": "heavy"}
    },
    
    "smart_conservative": {
        "description": "Smart balancing with conservative augmentation",
        "method": "balance_smart",
        "params": {"multiplier": 1.2, "augmentation_level": "light"}
    },
    
    "smart_aggressive": {
        "description": "Smart balancing with aggressive augmentation", 
        "method": "balance_smart",
        "params": {"multiplier": 2.0, "augmentation_level": "heavy"}
    },
    
    "custom_targets": {
        "description": "Custom target per class",
        "method": "balance_to_target",
        "params": {
            "targets": {
                "3": 1500, "4": 1500, "5": 1500, "6": 1500, "7": 1500,
                "8": 1500, "9": 1500, "18": 1200, "19": 1200, "20": 1200
            },
            "augmentation_level": "medium"
        }
    }
}

def quick_balance(source_dir: str, output_dir: str, config_name: str = "smart_conservative"):
    """
    Quick balancing with predefined configuration
    
    Args:
        source_dir: Source directory with class folders
        output_dir: Output directory
        config_name: Configuration name from BALANCE_CONFIGS
    """
    if config_name not in BALANCE_CONFIGS:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available configs: {list(BALANCE_CONFIGS.keys())}")
        return
    
    config = BALANCE_CONFIGS[config_name]
    print(f"🚀 Quick Balance: {config['description']}")
    
    balancer = SimpleDataBalancer(source_dir, output_dir)
    balancer.analyze_distribution()
    
    # Call the appropriate method
    method = getattr(balancer, config['method'])
    method(**config['params'])

def interactive_balance():
    """Interactive balancing with user input"""
    print("🎛️  Interactive Data Balancer")
    print("=" * 40)
    
    # Get source and output directories
    source = input("Source directory: ").strip()
    output = input("Output directory: ").strip()
    
    if not source or not output:
        print("❌ Please provide both source and output directories")
        return
    
    balancer = SimpleDataBalancer(source, output)
    distribution = balancer.analyze_distribution()
    
    if not distribution:
        print("❌ No classes found in source directory")
        return
    
    print(f"\nAvailable configurations:")
    for i, (name, config) in enumerate(BALANCE_CONFIGS.items(), 1):
        print(f"  {i}. {name}: {config['description']}")
    
    print(f"  {len(BALANCE_CONFIGS) + 1}. custom: Define your own targets")
    
    choice = input(f"\nSelect option (1-{len(BALANCE_CONFIGS) + 1}): ").strip()
    
    try:
        choice_idx = int(choice) - 1
        config_names = list(BALANCE_CONFIGS.keys())
        
        if 0 <= choice_idx < len(config_names):
            # Use predefined config
            config_name = config_names[choice_idx]
            config = BALANCE_CONFIGS[config_name]
            
            print(f"\n🎯 Using: {config['description']}")
            
            method = getattr(balancer, config['method'])
            method(**config['params'])
            
        elif choice_idx == len(config_names):
            # Custom targets
            print(f"\n📝 Define custom targets for each class:")
            targets = {}
            
            for class_name in distribution.keys():
                current = distribution[class_name]
                target = input(f"Class {class_name} (current: {current}): ").strip()
                if target:
                    targets[class_name] = int(target)
                else:
                    targets[class_name] = current
            
            aug_level = input("Augmentation level (light/medium/heavy): ").strip() or "medium"
            
            balancer.balance_to_target(targets, aug_level)
        
        else:
            print("❌ Invalid option")
    
    except (ValueError, IndexError):
        print("❌ Invalid option")

if __name__ == "__main__":
    print("🚀 Simple Data Balancer")
    print("=" * 30)
    
    # Example usage
    SOURCE_DIR = "classification"
    OUTPUT_DIR = "classification_balanced"
    
    # Method 1: Quick balance with predefined config
    print("Available quick configs:")
    for name, config in BALANCE_CONFIGS.items():
        print(f"  - {name}: {config['description']}")
    
    print(f"\nExample usage:")
    print(f"  quick_balance('{SOURCE_DIR}', '{OUTPUT_DIR}', 'smart_conservative')")
    print(f"  interactive_balance()")
    
    # Uncomment to run
    # quick_balance(SOURCE_DIR, OUTPUT_DIR, "smart_conservative")
    # interactive_balance()

# CLI usage examples:
"""
# Quick balancing
python simple_balancer.py

# Or import and use:
from simple_balancer import quick_balance, interactive_balance

# Quick methods
quick_balance("classification", "balanced_output", "equal_1000")
quick_balance("classification", "balanced_output", "smart_aggressive") 

# Interactive
interactive_balance()

# Manual control
balancer = SimpleDataBalancer("classification", "balanced_output")
balancer.analyze_distribution()
balancer.balance_equal(target_size=1500, augmentation_level="medium")
"""
