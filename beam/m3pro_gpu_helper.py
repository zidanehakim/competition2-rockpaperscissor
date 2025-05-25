"""
M3 Pro Mac GPU Helper for PyTorch

This script provides helper functions to detect and use Apple Silicon GPUs (M1/M2/M3) in PyTorch.
Import these functions in your notebook to enable GPU acceleration on M3 Pro Macs.
"""

import torch
import platform

def get_device():
    """
    Get the appropriate device for PyTorch based on availability.
    Checks for CUDA, MPS (Apple Silicon), or falls back to CPU.
    
    Returns:
        torch.device: The appropriate device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def setup_m3pro_gpu():
    """
    Set up PyTorch to use M3 Pro GPU if available.
    Returns the device and prints device information.
    
    Returns:
        torch.device: The best available device
    """
    device = get_device()
    
    # Print system and device information
    print(f"🖥️ System: {platform.system()} {platform.machine()}")
    
    if device.type == "cuda":
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 CUDA Version: {torch.version.cuda}")
        print(f"📋 Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device.type == "mps":
        print(f"🚀 Using Apple Silicon GPU via MPS")
        print(f"📊 PyTorch Version: {torch.__version__}")
    else:
        print(f"💻 Using CPU: PyTorch {torch.__version__}")
    
    return device

def seed_everything(seed=42, use_deterministic=True):
    """
    Set seeds for reproducibility, accounting for M3 Pro GPU.
    
    Args:
        seed (int): Seed for reproducibility
        use_deterministic (bool): Whether to enable deterministic algorithms
    """
    import random
    import numpy as np
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Set deterministic flag (note: may slow down performance)
    if use_deterministic:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # For MPS, some operations may not have deterministic implementations
        # so we don't set additional flags

# Usage examples to add to your notebook:
"""
# 1. At the top of your notebook, import the helper:
from m3pro_gpu_helper import setup_m3pro_gpu, seed_everything

# 2. Replace your device setup code with:
device = setup_m3pro_gpu()

# 3. Replace your seed function with:
seed_everything(42)
"""
