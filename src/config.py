import os
import torch

class Config:
    """Optimized configuration for high-end GPU"""
    
    # Paths
    # Default to the previously set path, but allow environment variable override 
    # to facilitate reproducibility across different machines in open-source settings.
    DATASET_PATH = os.environ.get("DATASET_PATH", r"D:\VB2\PD\Plant_leaf_diseases_dataset_without_augmentation")
    OUTPUT_BASE = "outputs"
    PROGRESS_FILE = "outputs/progress.json"
    CACHE_DIR = "outputs/preprocessed_cache"
    
    # Data split ratios
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    
    # Image parameters
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 512  # Large batch for high-end GPU
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 8
    
    # GPU Optimization - WINDOWS/JUPYTER
    MIXED_PRECISION = True
    USE_TORCH_COMPILE = False  # Disable for now (causes loading issues)
    NUM_WORKERS = 0  # Must be 0 on Windows with Jupyter
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False  # Must be False when num_workers=0
    PREFETCH_FACTOR = None  # Must be None when num_workers=0
    
    # Preprocessing cache
    USE_CACHE = True  # Cache preprocessed images to SSD
    
    # Segmentation parameters
    KMEANS_CLUSTERS = 3
    
    # Augmentation parameters
    STANDARD_AUG = {
        'rotation_range': 20,
        'horizontal_flip': True,
        'vertical_flip': True,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.1
    }
    
    HEAVY_AUG = {
        'rotation_range': 40,
        'horizontal_flip': True,
        'vertical_flip': True,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'zoom_range': 0.2,
        'brightness_range': [0.8, 1.2],
        'shear_range': 0.2
    }
    
    # Sample saving parameters
    SAMPLE_IMAGES_PER_STEP = 20
    GRADCAM_IMAGES_PER_CLASS = 10
    
    # Model configurations
    SEGMENTATION_METHODS = ['none', 'otsu', 'kmeans']
    INPUT_METHODS = ['raw', 'masking', 'cropping']
    AUGMENTATION_TYPES = ['standard', 'heavy']
    MODEL_TYPES = ['mobilenetv2', 'squeezenet']  
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
