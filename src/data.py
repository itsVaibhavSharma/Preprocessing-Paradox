import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.config import Config
from src.utils import get_class_names, save_metadata

class DatasetSplitter:
    """Split dataset with stratification"""
    
    def __init__(self, dataset_path, output_path, progress_tracker):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.progress_tracker = progress_tracker
        self.class_names = get_class_names(dataset_path)
        self.n_classes = len(self.class_names)
    
    def split_and_copy(self):
        """Split dataset"""
        
        if self.progress_tracker.is_dataset_split_done():
            print("\n" + "="*70)
            print("PHASE 0: DATASET SPLITTING (Already Complete)")
            print("="*70)
            try:
                with open(os.path.join(Config.OUTPUT_BASE, 'final_results', 'dataset_split_info.json'), 'r') as f:
                    import json
                    split_info = json.load(f)
                print(f"\n Using existing dataset split: {split_info['total_classes']} classes")
                return split_info
            except:
                pass
        
        print("\n" + "="*70)
        print("PHASE 0: DATASET SPLITTING")
        print("="*70)
        
        split_info = {
            'total_classes': self.n_classes,
            'class_names': self.class_names,
            'split_ratios': {
                'train': Config.TRAIN_RATIO,
                'val': Config.VAL_RATIO,
                'test': Config.TEST_RATIO
            },
            'class_distribution': {}
        }
        
        try:
            for class_idx, class_name in enumerate(tqdm(self.class_names, desc="Splitting classes")):
                class_path = os.path.join(self.dataset_path, class_name)
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                n_images = len(image_files)
                if n_images == 0:
                    continue
                
                n_train = int(n_images * Config.TRAIN_RATIO)
                n_val = int(n_images * Config.VAL_RATIO)
                n_test = n_images - n_train - n_val
                
                np.random.shuffle(image_files)
                train_files = image_files[:n_train]
                val_files = image_files[n_train:n_train + n_val]
                test_files = image_files[n_train + n_val:]
                
                for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                    split_class_path = os.path.join(self.output_path, split_name, class_name)
                    os.makedirs(split_class_path, exist_ok=True)
                    
                    for file in files:
                        try:
                            src = os.path.join(class_path, file)
                            dst = os.path.join(split_class_path, file)
                            shutil.copy2(src, dst)
                        except:
                            pass
                
                split_info['class_distribution'][class_name] = {
                    'total': n_images,
                    'train': n_train,
                    'val': n_val,
                    'test': n_test
                }
            
            save_metadata(split_info, 'dataset_split_info.json')
            self.progress_tracker.mark_dataset_split_done()
            
            print(f"\n Dataset Split Complete: {self.n_classes} classes")
            
        except Exception as e:
            print(f"Error during dataset splitting: {e}")
        
        return split_info


class ImagePreprocessor:
    """Optimized preprocessing with caching"""
    
    @staticmethod
    def apply_clahe(image):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        except:
            return image
    
    @staticmethod
    def segment_otsu(image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            return binary
        except:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    @staticmethod
    def segment_kmeans(image, k=3):
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            pixel_values = hsv.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, 
                                           cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            labels = labels.flatten()
            
            v_channel = hsv[:, :, 2]
            cluster_means = []
            for i in range(k):
                mask = (labels.reshape(image.shape[:2]) == i)
                if mask.sum() > 0:
                    cluster_means.append(np.mean(v_channel[mask]))
                else:
                    cluster_means.append(255)
            
            disease_cluster = np.argmin(cluster_means)
            binary_mask = (labels.reshape(image.shape[:2]) == disease_cluster).astype(np.uint8) * 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            return binary_mask
        except:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    @staticmethod
    def apply_masking(image, mask):
        try:
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            masked_image = cv2.bitwise_and(image, mask_3channel)
            return masked_image
        except:
            return image
    
    @staticmethod
    def apply_cropping(image, mask):
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return image
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            cropped = image[y:y+h, x:x+w]
            return cropped if cropped.size > 0 else image
        except:
            return image
    
    @staticmethod
    def preprocess_image(image_path, seg_method='none', input_method='raw'):
        """Preprocess image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((*Config.IMG_SIZE, 3), dtype=np.uint8)
            
            image = cv2.resize(image, Config.IMG_SIZE)
            enhanced = ImagePreprocessor.apply_clahe(image)
            
            if seg_method == 'none':
                return enhanced
            
            if seg_method == 'otsu':
                mask = ImagePreprocessor.segment_otsu(enhanced)
            elif seg_method == 'kmeans':
                mask = ImagePreprocessor.segment_kmeans(enhanced, k=Config.KMEANS_CLUSTERS)
            else:
                return enhanced
            
            if input_method == 'masking':
                processed = ImagePreprocessor.apply_masking(enhanced, mask)
            elif input_method == 'cropping':
                processed = ImagePreprocessor.apply_cropping(enhanced, mask)
                processed = cv2.resize(processed, Config.IMG_SIZE)
            else:
                processed = enhanced
            
            return processed
        except:
            return np.zeros((*Config.IMG_SIZE, 3), dtype=np.uint8)

class PreprocessingCache:
    """Cache preprocessed images to disk for fast loading"""
    
    def __init__(self, cache_dir, seg_method, input_method):
        self.cache_dir = os.path.join(cache_dir, f"{seg_method}_{input_method}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.seg_method = seg_method
        self.input_method = input_method
    
    def get_cache_path(self, image_path):
        """Get cache path for an image"""
        rel_path = image_path.replace('\\', '_').replace('/', '_').replace(':', '')
        return os.path.join(self.cache_dir, f"{rel_path}.npy")
    
    def exists(self, image_path):
        """Check if cached version exists"""
        return os.path.exists(self.get_cache_path(image_path))
    
    def load(self, image_path):
        """Load cached preprocessed image"""
        try:
            return np.load(self.get_cache_path(image_path))
        except:
            return None
    
    def save(self, image_path, image):
        """Save preprocessed image to cache"""
        try:
            np.save(self.get_cache_path(image_path), image)
        except:
            pass

class CustomAugmentation:
    """Fast augmentation"""
    
    def __init__(self, aug_config=None):
        self.aug_config = aug_config if aug_config is not None else {}
    
    def __call__(self, image):
        try:
            if not self.aug_config:
                return image
            
            if self.aug_config.get('horizontal_flip', False) and np.random.random() > 0.5:
                image = cv2.flip(image, 1)
            
            if self.aug_config.get('vertical_flip', False) and np.random.random() > 0.5:
                image = cv2.flip(image, 0)
            
            if 'rotation_range' in self.aug_config:
                angle = np.random.uniform(-self.aug_config['rotation_range'], 
                                         self.aug_config['rotation_range'])
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            if 'brightness_range' in self.aug_config:
                factor = np.random.uniform(self.aug_config['brightness_range'][0],
                                          self.aug_config['brightness_range'][1])
                image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            if 'zoom_range' in self.aug_config:
                zoom_factor = np.random.uniform(1 - self.aug_config['zoom_range'],
                                               1 + self.aug_config['zoom_range'])
                h, w = image.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                resized = cv2.resize(image, (new_w, new_h))
                
                if zoom_factor > 1:  # Zoom in - crop center
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    image = resized[start_h:start_h+h, start_w:start_w+w]
                else:  # Zoom out - pad
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    image = cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, 
                                              pad_w, w-new_w-pad_w, 
                                              cv2.BORDER_REFLECT)
            
            if 'shear_range' in self.aug_config:
                shear = np.random.uniform(-self.aug_config['shear_range'],
                                         self.aug_config['shear_range'])
                h, w = image.shape[:2]
                M = np.float32([[1, shear, 0], [0, 1, 0]])
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            return image
        except:
            return image

class OptimizedPlantDiseaseDataset(Dataset):
    """GPU-optimized dataset with caching"""
    
    def __init__(self, image_paths, labels, seg_method='none', 
                 input_method='raw', augmentation=None, use_cache=True):
        self.image_paths = image_paths
        self.labels = labels
        self.seg_method = seg_method
        self.input_method = input_method
        self.augmentation = augmentation
        self.use_cache = use_cache and Config.USE_CACHE
        
        # Initialize cache
        if self.use_cache:
            self.cache = PreprocessingCache(Config.CACHE_DIR, seg_method, input_method)
        
        # PyTorch transforms
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Try to load from cache first
            if self.use_cache and self.cache.exists(img_path):
                image = self.cache.load(img_path)
            else:
                # Preprocess image
                image = ImagePreprocessor.preprocess_image(
                    img_path, self.seg_method, self.input_method
                )
                # Save to cache
                if self.use_cache:
                    self.cache.save(img_path, image)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            if self.augmentation is not None:
                image = self.augmentation(image)
            
            # Convert to tensor
            image = self.to_tensor(image)
            
            return image, label
        except:
            blank = torch.zeros(3, *Config.IMG_SIZE)
            return blank, self.labels[idx]

def create_optimized_data_loaders(split_data_path, seg_method, input_method, aug_type):
    """Create optimized data loaders"""
    
    def get_data(split):
        paths, labels = [], []
        split_path = os.path.join(split_data_path, split)
        
        if not os.path.exists(split_path):
            return paths, labels
        
        class_names = sorted(os.listdir(split_path))
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            paths.extend(images)
            labels.extend([class_idx] * len(images))
        
        return paths, labels
    
    train_paths, train_labels = get_data('train')
    val_paths, val_labels = get_data('val')
    test_paths, test_labels = get_data('test')
    
    # Create augmentation
    if aug_type == 'standard':
        aug_config = Config.STANDARD_AUG
    elif aug_type == 'heavy':
        aug_config = Config.HEAVY_AUG
    else:
        aug_config = None
    
    augmentation = CustomAugmentation(aug_config) if aug_config else None
    
    # Create datasets
    train_dataset = OptimizedPlantDiseaseDataset(
        train_paths, train_labels, seg_method, input_method, augmentation, use_cache=True
    )
    
    val_dataset = OptimizedPlantDiseaseDataset(
        val_paths, val_labels, seg_method, input_method, None, use_cache=True
    )
    
    test_dataset = OptimizedPlantDiseaseDataset(
        test_paths, test_labels, seg_method, input_method, None, use_cache=True
    )
    
    loader_kwargs = {
        'batch_size': Config.BATCH_SIZE,
        'pin_memory': Config.PIN_MEMORY,
    }
    
    # Only add these params if num_workers > 0
    if Config.NUM_WORKERS > 0:
        loader_kwargs['num_workers'] = Config.NUM_WORKERS
        loader_kwargs['persistent_workers'] = Config.PERSISTENT_WORKERS
        loader_kwargs['prefetch_factor'] = Config.PREFETCH_FACTOR
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader, (test_paths, test_labels)


class PreprocessingSampleSaver:
    """Save sample preprocessed and augmented images"""
    
    def __init__(self, split_data_path, progress_tracker):
        self.split_data_path = split_data_path
        self.progress_tracker = progress_tracker
        self.class_names = None
    
    def save_preprocessing_samples(self):
        """Save sample images showing preprocessing steps"""
        
        if self.progress_tracker.is_preprocessing_done():
            print("\n" + "="*70)
            print("PHASE 1: PREPROCESSING SAMPLES (Already Complete)")
            print("="*70)
            return
        
        print("\n" + "="*70)
        print("PHASE 1: SAVING PREPROCESSING SAMPLES")
        print("="*70)
        
        train_path = os.path.join(self.split_data_path, 'train')
        if not os.path.exists(train_path):
            return
        
        self.class_names = sorted(os.listdir(train_path))
        if len(self.class_names) == 0:
            return
        
        # Get sample images from first class
        first_class_path = os.path.join(train_path, self.class_names[0])
        image_files = [f for f in os.listdir(first_class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            return
        
        # Sample images to process
        num_samples = min(Config.SAMPLE_IMAGES_PER_STEP, len(image_files))
        sample_files = np.random.choice(image_files, num_samples, replace=False)
        
        print(f"\nSaving {num_samples} preprocessing samples...")
        
        for idx, img_file in enumerate(tqdm(sample_files, desc="Processing samples")):
            try:
                img_path = os.path.join(first_class_path, img_file)
                original = cv2.imread(img_path)
                if original is None:
                    continue
                
                original = cv2.resize(original, Config.IMG_SIZE)
                
                # CLAHE Enhancement
                clahe_img = ImagePreprocessor.apply_clahe(original)
                save_path = os.path.join(Config.OUTPUT_BASE, 'preprocessed_samples', 
                                        'clahe', f'sample_{idx:03d}.png')
                cv2.imwrite(save_path, clahe_img)
                
                # Otsu Segmentation
                otsu_mask = ImagePreprocessor.segment_otsu(clahe_img)
                save_path = os.path.join(Config.OUTPUT_BASE, 'preprocessed_samples', 
                                        'otsu', f'sample_{idx:03d}.png')
                cv2.imwrite(save_path, otsu_mask)
                
                # KMeans Segmentation
                kmeans_mask = ImagePreprocessor.segment_kmeans(clahe_img)
                save_path = os.path.join(Config.OUTPUT_BASE, 'preprocessed_samples', 
                                        'kmeans', f'sample_{idx:03d}.png')
                cv2.imwrite(save_path, kmeans_mask)
                
                # Masking
                masked = ImagePreprocessor.apply_masking(clahe_img, otsu_mask)
                save_path = os.path.join(Config.OUTPUT_BASE, 'preprocessed_samples', 
                                        'masking', f'sample_{idx:03d}.png')
                cv2.imwrite(save_path, masked)
                
                # Cropping
                cropped = ImagePreprocessor.apply_cropping(clahe_img, otsu_mask)
                cropped_resized = cv2.resize(cropped, Config.IMG_SIZE)
                save_path = os.path.join(Config.OUTPUT_BASE, 'preprocessed_samples', 
                                        'cropping', f'sample_{idx:03d}.png')
                cv2.imwrite(save_path, cropped_resized)
                
            except Exception as e:
                continue
        
        # Save augmentation samples
        self.save_augmentation_samples(sample_files[:5], first_class_path)
        
        self.progress_tracker.mark_preprocessing_done()
        print(" Preprocessing samples saved!")
    
    def save_augmentation_samples(self, sample_files, class_path):
        """Save augmented samples for both standard and heavy augmentation"""
        
        print("\nSaving augmentation samples...")
        
        for aug_type in ['standard', 'heavy']:
            aug_dir = os.path.join(Config.OUTPUT_BASE, 'preprocessed_samples', 
                                  'augmentation', aug_type)
            os.makedirs(aug_dir, exist_ok=True)
            
            if aug_type == 'standard':
                aug_config = Config.STANDARD_AUG
            else:
                aug_config = Config.HEAVY_AUG
            
            augmentor = CustomAugmentation(aug_config)
            
            for idx, img_file in enumerate(sample_files):
                try:
                    img_path = os.path.join(class_path, img_file)
                    original = cv2.imread(img_path)
                    if original is None:
                        continue
                    
                    original = cv2.resize(original, Config.IMG_SIZE)
                    
                    # Create comparison figure: Original + 5 augmented versions
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Original
                    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                    axes[0, 0].set_title('Original', fontsize=12)
                    axes[0, 0].axis('off')
                    
                    # 5 augmented versions
                    for aug_idx in range(5):
                        augmented = augmentor(original.copy())
                        row = (aug_idx + 1) // 3
                        col = (aug_idx + 1) % 3
                        axes[row, col].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
                        axes[row, col].set_title(f'Augmented {aug_idx+1}', fontsize=12)
                        axes[row, col].axis('off')
                    
                    plt.suptitle(f'{aug_type.capitalize()} Augmentation Examples', 
                               fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    save_path = os.path.join(aug_dir, f'comparison_{idx:03d}.png')
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    continue
        
        print(f" Augmentation samples saved!")
