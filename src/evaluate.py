import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms

from src.config import Config
from src.data import ImagePreprocessor, create_optimized_data_loaders
from src.models import build_optimized_model

class GradCAM:
    """Grad-CAM for visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image, class_idx):
        try:
            self.model.eval()
            image = image.unsqueeze(0).to(Config.DEVICE)
            output = self.model(image)
            
            self.model.zero_grad()
            class_score = output[0, class_idx]
            class_score.backward()
            
            gradients = self.gradients[0].cpu().numpy()
            activations = self.activations[0].cpu().numpy()
            
            weights = np.mean(gradients, axis=(1, 2))
            heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
            
            for i, w in enumerate(weights):
                heatmap += w * activations[i]
            
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            heatmap = cv2.resize(heatmap, Config.IMG_SIZE)
            heatmap = np.uint8(255 * heatmap)
            
            return heatmap
        except:
            return np.zeros(Config.IMG_SIZE, dtype=np.uint8)
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        try:
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            if image.shape[:2] != Config.IMG_SIZE:
                image = cv2.resize(image, Config.IMG_SIZE)
            
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
            
            return superimposed
        except:
            return image

def get_target_layer(model, model_type):
    try:
        if model_type == 'squeezenet':
            return model.features[-1]
        elif model_type == 'mobilenetv2':
            return model.features[-1]
    except:
        raise

class ModelEvaluator:
    """Model evaluation with Grad-CAM"""
    
    def __init__(self, split_data_path, class_names, progress_tracker):
        self.split_data_path = split_data_path
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.progress_tracker = progress_tracker
        self.device = Config.DEVICE
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("PHASE 3: MODEL EVALUATION")
        print("="*70)
        
        model_dir = os.path.join(Config.OUTPUT_BASE, 'models')
        if not os.path.exists(model_dir):
            return
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if len(model_files) == 0:
            return
        
        evaluated_models = self.progress_tracker.get_evaluated_models()
        remaining_models = [f for f in model_files if f.replace('.pth', '') not in evaluated_models]
        
        if len(remaining_models) == 0:
            print("\n All models already evaluated!")
            return
        
        print(f"Total: {len(model_files)} | Evaluated: {len(evaluated_models)} | Remaining: {len(remaining_models)}\n")
        
        for model_file in tqdm(remaining_models, desc="Evaluating"):
            model_name = model_file.replace('.pth', '')
            
            try:
                config = self.parse_model_name(model_name)
                model = build_optimized_model(config['model_type'], self.n_classes)
                model.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
                model = model.to(self.device)
                model.eval()
                
                _, _, test_loader, test_data = create_optimized_data_loaders(
                    self.split_data_path,
                    config['seg_method'],
                    config['input_method'],
                    config['aug_type']
                )
                
                self.generate_confusion_matrix(model, test_loader, model_name)
                self.generate_gradcam_visualizations(model, test_data, model_name, config)
                
                self.progress_tracker.mark_model_evaluated(model_name)
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        print("\n Evaluation complete!")
    
    def parse_model_name(self, model_name):
        try:
            parts = model_name.split('_')
            
            if model_name.startswith('Baseline'):
                return {
                    'seg_method': 'none',
                    'input_method': 'raw',
                    'aug_type': parts[-2],
                    'model_type': parts[-1]
                }
            else:
                seg_input = parts[0].replace('SG-', '')
                seg_parts = seg_input.split('-')
                
                return {
                    'seg_method': seg_parts[0].lower(),
                    'input_method': seg_parts[1].lower() if len(seg_parts) > 1 else 'raw',
                    'aug_type': parts[-2],
                    'model_type': parts[-1]
                }
        except:
            return {'seg_method': 'none', 'input_method': 'raw', 'aug_type': 'standard', 'model_type': 'mobilenetv2'}
    
    def generate_confusion_matrix(self, model, test_loader, model_name):
        try:
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    try:
                        images = images.to(self.device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        
                        y_true.extend(labels.numpy())
                        y_pred.extend(predicted.cpu().numpy())
                    except:
                        continue
            
            if len(y_true) == 0:
                return
            
            cm = confusion_matrix(y_true, y_pred)
            
            cm_path = os.path.join(Config.OUTPUT_BASE, 'confusion_matrices', f'{model_name}_cm.npy')
            np.save(cm_path, cm)
            
            plt.figure(figsize=(20, 18))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            
            save_path = os.path.join(Config.OUTPUT_BASE, 'confusion_matrices', f'{model_name}_cm.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                          output_dict=True, zero_division=0)
            report_path = os.path.join(Config.OUTPUT_BASE, 'metrics', f'{model_name}_classification_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
    
    def generate_gradcam_visualizations(self, model, test_data, model_name, config):
        try:
            test_paths, test_labels = test_data
            
            if len(test_paths) == 0:
                return
            
            target_layer = get_target_layer(model, config['model_type'])
            gradcam = GradCAM(model, target_layer)
            
            images_per_class = Config.GRADCAM_IMAGES_PER_CLASS
            
            output_dir = os.path.join(Config.OUTPUT_BASE, 'gradcam', model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            processed_count = 0
            
            for class_idx, class_name in enumerate(self.class_names):
                class_images = [(p, l) for p, l in zip(test_paths, test_labels) if l == class_idx]
                
                if len(class_images) == 0:
                    continue
                
                sampled = class_images[:images_per_class]
                
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                for img_idx, (img_path, true_label) in enumerate(sampled):
                    try:
                        processed_img = ImagePreprocessor.preprocess_image(
                            img_path, config['seg_method'], config['input_method']
                        )
                        
                        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = transform(img_rgb)
                        
                        with torch.no_grad():
                            output = model(img_tensor.unsqueeze(0).to(self.device))
                            pred_class = output.argmax(1).item()
                        
                        heatmap = gradcam.generate_heatmap(img_tensor, pred_class)
                        superimposed = gradcam.overlay_heatmap(img_rgb, heatmap)
                        
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        axes[0].imshow(img_rgb)
                        axes[0].set_title('Original')
                        axes[0].axis('off')
                        
                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title('Heatmap')
                        axes[1].axis('off')
                        
                        axes[2].imshow(superimposed)
                        axes[2].set_title(f'Pred: {self.class_names[pred_class]}')
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        
                        save_path = os.path.join(class_dir, f'gradcam_{img_idx:03d}.png')
                        plt.savefig(save_path, dpi=100, bbox_inches='tight')
                        plt.close()
                        
                        processed_count += 1
                        
                    except:
                        continue
            
            print(f"  Generated {processed_count} Grad-CAM visualizations")
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
