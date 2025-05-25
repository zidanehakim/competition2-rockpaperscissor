# Beam Damage Multi-Header Classification

This project implements a multi-task deep learning model for classifying structural beam images into their primary class categories while simultaneously identifying various types of damage present in the beams.

## Project Overview

The task involves classifying beam images into three main structural classes (A, B, C) while simultaneously detecting multiple types of damage that may be present in each beam. This is implemented as a multi-task learning problem with:

1. A shared backbone (EfficientNet) that extracts features from images
2. Two separate classification heads:
   - Class head: Identifies the primary beam structure type (3 classes)
   - Damage head: Detects multiple possible damage types (11 categories)

## Dataset

The dataset consists of beam images labeled with:
- `class_label`: The primary class of the beam (18, 19, or 20, corresponding to Classes A, B, and C)
- `damage_labels`: A comma-separated list of damage types present in the image (labels 0-10)

Each class is associated with specific allowed damage types:
- Class A (18): Damage type 0
- Class B (19): Damage types 3, 4, 6, 8
- Class C (20): Damage types 1, 5, 7, 9, 10

## Notebook Structure

The `multiheader-classification.ipynb` notebook contains the complete pipeline for this project. Here's a detailed breakdown of each cell:

### Cell 1: Data Preparation

This cell handles:
- Loading the original dataset from CSV
- Splitting into training and validation sets (80/20 split)
- Analyzing the distribution of damage labels
- Identifying rare damage labels (those with <50 occurrences)
- Oversampling rare damage categories to balance the dataset
- Saving the processed datasets to CSV files

The oversampling strategy duplicates images with rare damage labels to ensure the model has sufficient examples to learn from.

### Cell 2: Model Training

This cell contains:

1. **Imports and Setup**:
   - Essential libraries (PyTorch, OpenCV, Albumentations, etc.)
   - Random seed setting for reproducibility
   - GPU detection and setup (supports CUDA and Apple Silicon MPS)

2. **Dataset Implementation**:
   - Custom `AlbumentationsDamageDataset` class
   - Image loading and preprocessing
   - Label handling for both classification tasks

3. **Model Architecture**:
   - `MultiTaskDamageModel` class based on EfficientNet (B3, B4, or B5 variants)
   - Shared backbone with two separate task heads

4. **Data Augmentation**:
   - Training transforms (random crops, flips, brightness adjustments, etc.)
   - Validation transforms (resizing and normalization)

5. **Training Configuration**:
   - Hyperparameters (batch size, learning rate, etc.)
   - Loss functions (weighted BCE for damage labels to handle class imbalance)
   - Optimizer setup

6. **Training Loop**:
   - Epoch-based training with validation
   - Metrics tracking (accuracy, F1 score for both tasks)
   - Model checkpoint saving based on best validation damage F1 score

The training process leverages mixed precision where available for faster training on modern GPUs.

### Cell 3: Inference and Submission

This final cell handles:
- Loading the trained model
- Running inference on test images
- Applying class-specific damage label filtering
- Testing different confidence thresholds
- Generating submission files in the required format

The inference process ensures that only the allowed damage types for each predicted class are considered, improving the accuracy of the final predictions.

## Key Features

- **Multi-task Learning**: Jointly training for beam class and damage detection
- **Class-specific Damage Filtering**: Only allows valid damage types for each beam class
- **Oversampling Strategy**: Balances rare damage classes
- **Hardware Acceleration**: Supports both NVIDIA (CUDA) and Apple Silicon (MPS) GPUs
- **Threshold Tuning**: Tests multiple confidence thresholds to optimize results

## Hardware Requirements

- GPU recommended (CUDA or Apple Silicon MPS supported)
- Minimum 8GB RAM
- Storage for datasets and model checkpoints

## Usage

1. Prepare your dataset in the required CSV format
2. Run the data preparation cell to create balanced training data
3. Run the training cell to train the model (adjust hyperparameters as needed)
4. Run the inference cell to generate predictions

## Results

The model achieves strong performance on both tasks:
- High accuracy in classifying beam types
- Effective detection of multiple damage types
- F1 score optimization through threshold tuning

## Extensions

Potential improvements to explore:
- Ensemble methods combining multiple EfficientNet variants
- More sophisticated data augmentation techniques
- Test-time augmentation for improved inference
- Deeper exploration of loss weighting strategies

## Supporting Modules

This project includes several supporting modules that enhance the main notebook functionality:

### `m3pro_gpu_helper.py`

This module provides hardware acceleration support for Apple Silicon Macs (M1/M2/M3 series) in PyTorch. Its key functions include:

- **`get_device()`**: Automatically detects the best available computing device, prioritizing in this order:
  1. CUDA (for NVIDIA GPUs)
  2. MPS (Metal Performance Shaders for Apple Silicon GPUs)
  3. CPU (as fallback)

- **`setup_m3pro_gpu()`**: Configures PyTorch to use the M3 Pro GPU if available, and provides detailed system information about the detected hardware

- **`seed_everything()`**: Sets random seeds for reproducibility across different hardware platforms, with special considerations for the M3 Pro GPU

This module ensures your deep learning model can leverage hardware acceleration on modern Mac computers with M-series chips, which can provide significant speed improvements over CPU training.

### `labelme_custom.py`

This is a custom image labeling tool built with Tkinter, specifically designed for this beam damage classification project. Key features include:

- Provides a graphical user interface for annotating beam images
- Maps beam classes (A, B, C) to their numerical labels (18, 19, 20)
- Implements class-specific damage type options:
  - Class A: Exposed rebar (0)
  - Class B: Various continuous crack types (3, 4, 6, 8)
  - Class C: No significant damage (1) and various discontinuous crack types (5, 7, 9, 10)
- Saves annotation results to a CSV file

This tool was used to create and enhance the dataset by manually labeling beam images with their correct class and damage types.

### `renaming.py`

This utility module handles file organization and standardization:

- Manages file renaming operations for the dataset
- Standardizes image filenames for consistency
- Converts between different naming conventions for the beam images
- Organizes images into the correct class folders

This preprocessing utility ensures all files follow a consistent naming pattern before being used in the training pipeline.
