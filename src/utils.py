import os
import json
import time
import shutil

from src.config import Config

class ProgressTracker:
    """Enhanced progress tracker with epoch-level resume"""
    
    def __init__(self, progress_file=Config.PROGRESS_FILE):
        self.progress_file = progress_file
        self.progress = self.load_progress()
    
    def load_progress(self):
        """Load existing progress"""
        # Default structure
        default_progress = {
            'dataset_split': False,
            'preprocessing_samples': False,
            'preprocessing_cache': {},
            'trained_models': [],
            'model_training_state': {},  # Epoch-level state
            'evaluated_models': [],
            'completed_phases': []
        }
        
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    loaded_progress = json.load(f)
                
                # Merge loaded progress with defaults to ensure all keys exist
                for key in default_progress:
                    if key not in loaded_progress:
                        loaded_progress[key] = default_progress[key]
                
                return loaded_progress
            except:
                pass
        
        return default_progress
    
    def save_progress(self, retries=5, delay=1):
        """Save progress with retry logic and atomic write"""
        for i in range(retries):
            try:
                os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
                
                # Write to a temporary file first
                temp_file = self.progress_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(self.progress, f, indent=4)
                
                # Atomically rename the temp file to the real file
                shutil.move(temp_file, self.progress_file) 
                
                return  # Success!
            
            except Exception as e:
                print(f"Warning: Could not save progress (Attempt {i+1}/{retries}): {e}")
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    print("\n" + "="*70)
                    print("CRITICAL ERROR: Failed to save progress file after retries.")
                    print("Please check file permissions, disk space, or antivirus locks.")
                    print(f"File: {os.path.abspath(self.progress_file)}")
                    print("Exiting to prevent progress loss.")
                    print("="*70)
                    raise
    
    def mark_dataset_split_done(self):
        self.progress['dataset_split'] = True
        self.save_progress()
    
    def mark_preprocessing_done(self):
        self.progress['preprocessing_samples'] = True
        self.save_progress()
    
    def mark_cache_complete(self, cache_key):
        self.progress['preprocessing_cache'][cache_key] = True
        self.save_progress()
    
    def is_cache_complete(self, cache_key):
        return self.progress['preprocessing_cache'].get(cache_key, False)
    
    def mark_model_trained(self, model_name):
        if model_name not in self.progress['trained_models']:
            self.progress['trained_models'].append(model_name)
            # Remove training state once complete
            if model_name in self.progress['model_training_state']:
                del self.progress['model_training_state'][model_name]
            self.save_progress()
    
    def save_training_state(self, model_name, epoch, best_val_loss):
        """Save training state for epoch-level resume"""
        self.progress['model_training_state'][model_name] = {
            'last_epoch': epoch,
            'best_val_loss': best_val_loss
        }
        self.save_progress()
    
    def get_training_state(self, model_name):
        """Get training state for resume"""
        return self.progress['model_training_state'].get(model_name, None)
    
    def mark_model_evaluated(self, model_name):
        if model_name not in self.progress['evaluated_models']:
            self.progress['evaluated_models'].append(model_name)
            self.save_progress()
    
    def is_dataset_split_done(self):
        return self.progress['dataset_split']
    
    def is_preprocessing_done(self):
        return self.progress['preprocessing_samples']
    
    def is_model_trained(self, model_name):
        return model_name in self.progress['trained_models']
    
    def is_model_evaluated(self, model_name):
        return model_name in self.progress['evaluated_models']
    
    def get_trained_models(self):
        return self.progress['trained_models']
    
    def get_evaluated_models(self):
        return self.progress['evaluated_models']

def create_directory_structure():
    """Create all necessary directories"""
    dirs = [
        'outputs',
        'outputs/split_data/train',
        'outputs/split_data/val',
        'outputs/split_data/test',
        'outputs/preprocessed_samples',
        'outputs/preprocessed_samples/clahe',
        'outputs/preprocessed_samples/otsu',
        'outputs/preprocessed_samples/kmeans',
        'outputs/preprocessed_samples/masking',
        'outputs/preprocessed_samples/cropping',
        'outputs/preprocessed_samples/augmentation',
        'outputs/preprocessed_cache',
        'outputs/models',
        'outputs/checkpoints',
        'outputs/training_history',
        'outputs/training_curves',
        'outputs/confusion_matrices',
        'outputs/gradcam',
        'outputs/metrics',
        'outputs/final_results'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f" Created directory structure at: {os.path.abspath('outputs')}")

def get_class_names(dataset_path):
    """Get all class names from dataset"""
    try:
        class_names = sorted([d for d in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, d))])
        return class_names
    except Exception as e:
        print(f"Error reading class names: {e}")
        return []

def save_metadata(metadata, filename):
    """Save metadata to JSON file"""
    try:
        filepath = os.path.join(Config.OUTPUT_BASE, 'final_results', filename)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save metadata: {e}")
