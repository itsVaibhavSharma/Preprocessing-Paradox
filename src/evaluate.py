import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms
from statsmodels.stats.contingency_tables import mcnemar

from src.config import Config
from src.data import ImagePreprocessor, create_optimized_data_loaders, OptimizedPlantDiseaseDataset, CustomAugmentation
from src.models import build_optimized_model
from torch.utils.data import DataLoader

class ExpectedCalibrationError:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def calculate(self, confidences, predictions, labels):
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = np.zeros(1)
        accs = []
        confs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
            prop_in_bin = in_bin.mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = predictions[in_bin] == labels[in_bin]
                avg_confidence_in_bin = confidences[in_bin].mean()
                avg_accuracy_in_bin = accuracy_in_bin.mean()
                ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
                accs.append(avg_accuracy_in_bin)
                confs.append(avg_confidence_in_bin)
            else:
                accs.append(0)
                confs.append(0)
                
        return ece.item(), accs, confs

class GradCAM:
    """Grad-CAM and Grad-CAM++ for visualization"""
    
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
    
    def generate_heatmap(self, image, class_idx, use_plus_plus=False):
        try:
            self.model.eval()
            image = image.unsqueeze(0).to(Config.DEVICE)
            output = self.model(image)
            
            self.model.zero_grad()
            class_score = output[0, class_idx]
            class_score.backward()
            
            gradients = self.gradients[0].cpu().numpy()
            activations = self.activations[0].cpu().numpy()
            
            if use_plus_plus:
                # Grad-CAM++ logic
                gradients_power_2 = gradients ** 2
                gradients_power_3 = gradients_power_2 * gradients
                sum_activations = np.sum(activations, axis=(1, 2))
                
                eps = 1e-7
                alpha = gradients_power_2 / (2 * gradients_power_2 + sum_activations[:, None, None] * gradients_power_3 + eps)
                weights = np.sum(alpha * np.maximum(gradients, 0), axis=(1, 2))
            else:
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
        elif model_type == 'shufflenetv2':
            return model.conv5
        elif model_type == 'mobilenetv3':
            return model.features[-1]
        elif model_type == 'efficientnet_b0':
            return model.features[-1]
    except:
        return list(model.children())[-2]

class ModelEvaluator:
    """Model evaluation with Grad-CAM, McNemar's, ECE, and domain shifts"""
    
    def __init__(self, split_data_path, class_names, progress_tracker):
        self.split_data_path = split_data_path
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.progress_tracker = progress_tracker
        self.device = Config.DEVICE
        self.ece_calc = ExpectedCalibrationError()
        
    def get_test_predictions(self, model, test_loader):
        y_true = []
        y_pred = []
        confidences = []
        with torch.no_grad():
            for images, labels in test_loader:
                try:
                    images = images.to(self.device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    conf, predicted = probs.max(1)
                    
                    y_true.extend(labels.numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    confidences.extend(conf.cpu().numpy())
                except:
                    continue
        return np.array(y_true), np.array(y_pred), np.array(confidences)
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("PHASE 3: MODEL EVALUATION & EXPLAINABILITY")
        print("="*70)
        
        model_dir = os.path.join(Config.OUTPUT_BASE, 'models')
        if not os.path.exists(model_dir):
            return
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and '_seed' not in f]
        if len(model_files) == 0: return
        
        evaluated_models = self.progress_tracker.get_evaluated_models()
        remaining_models = [f for f in model_files if f.replace('.pth', '') not in evaluated_models]
        
        if len(remaining_models) == 0:
            print("\n All models already evaluated!")
            return
            
        print(f"Total: {len(model_files)} | Evaluated: {len(evaluated_models)} | Remaining: {len(remaining_models)}\n")
        
        # Store predictions for McNemar's Test
        all_predictions = {}
        
        for model_file in tqdm(remaining_models, desc="Evaluating"):
            model_name = model_file.replace('.pth', '')
            
            try:
                config = self.parse_model_name(model_name)
                model = build_optimized_model(config['model_type'], self.n_classes)
                model.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
                model = model.to(self.device)
                model.eval()
                
                _, _, test_loader, test_data = create_optimized_data_loaders(
                    self.split_data_path, config['seg_method'], config['input_method'], config['aug_type']
                )
                
                # Base predictions
                y_true, y_pred, confidences = self.get_test_predictions(model, test_loader)
                all_predictions[model_name] = (y_true, y_pred)
                
                # Confusion Matrix
                self.generate_confusion_matrix(y_true, y_pred, model_name)
                
                # Calibration (ECE)
                self.generate_reliability_diagram(confidences, y_pred, y_true, model_name)
                
                # Domain Shifts (External validation equivalent)
                self.evaluate_domain_shifts(model, config, test_data, model_name)
                
                # Grad-CAM and quantitative explainability
                self.generate_gradcam_visualizations(model, test_data, model_name, config)
                
                self.progress_tracker.mark_model_evaluated(model_name)
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
                
        # Perform McNemar's test for all segmented models against their baseline
        self.perform_mcnemars_tests(all_predictions)
        
        print("\n Evaluation complete!")
        
    def perform_mcnemars_tests(self, all_predictions):
        print("\nPerforming pairwise McNemar's tests...")
        results = {}
        for model_name, (y_true_seg, y_pred_seg) in all_predictions.items():
            if 'Baseline' not in model_name:
                config = self.parse_model_name(model_name)
                baseline_name = f"Baseline_{config['aug_type']}_{config['model_type']}"
                if baseline_name in all_predictions:
                    y_true_base, y_pred_base = all_predictions[baseline_name]
                    
                    if len(y_true_base) != len(y_true_seg):
                        continue
                        
                    # Contingency table
                    base_correct = (y_pred_base == y_true_base)
                    seg_correct = (y_pred_seg == y_true_seg)
                    
                    c00 = np.sum((~base_correct) & (~seg_correct))
                    c01 = np.sum((~base_correct) & seg_correct)
                    c10 = np.sum(base_correct & (~seg_correct))
                    c11 = np.sum(base_correct & seg_correct)
                    
                    table = [[c11, c10], [c01, c00]]
                    stat = mcnemar(table, exact=False, correction=True)
                    
                    results[f"{model_name}_vs_{baseline_name}"] = {
                        "p_value": stat.pvalue,
                        "statistic": stat.statistic
                    }
        
        with open(os.path.join(Config.OUTPUT_BASE, 'metrics', 'mcnemars_test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    def evaluate_domain_shifts(self, model, config, test_data, model_name):
        test_paths, test_labels = test_data
        shifts_results = {}
        
        for shift in Config.DOMAIN_SHIFTS:
            aug_config = {f'domain_shift_{shift}': True}
            augmentation = CustomAugmentation(aug_config)
            
            shift_dataset = OptimizedPlantDiseaseDataset(
                test_paths, test_labels, config['seg_method'], config['input_method'], 
                augmentation, use_cache=False
            )
            shift_loader = DataLoader(shift_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            
            y_true, y_pred, _ = self.get_test_predictions(model, shift_loader)
            if len(y_true) > 0:
                acc = np.mean(y_true == y_pred)
                shifts_results[shift] = float(acc)
                
        with open(os.path.join(Config.OUTPUT_BASE, 'metrics', f'{model_name}_domain_shifts.json'), 'w') as f:
            json.dump(shifts_results, f, indent=4)
    
    def generate_reliability_diagram(self, confidences, y_pred, y_true, model_name):
        ece, accs, confs = self.ece_calc.calculate(confidences, y_pred, y_true)
        
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.bar(np.linspace(0.05, 0.95, 15), accs, width=0.1, alpha=0.5, edgecolor='black', label='Model Accuracy')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram\n{model_name} (ECE: {ece:.4f})')
        plt.legend()
        
        os.makedirs(os.path.join(Config.OUTPUT_BASE, 'metrics'), exist_ok=True)
        plt.savefig(os.path.join(Config.OUTPUT_BASE, 'metrics', f'{model_name}_reliability.png'))
        plt.close()
        
    def parse_model_name(self, model_name):
        try:
            # Temporarily replace efficientnet_b0 to avoid split issues
            model_name = model_name.replace('efficientnet_b0', 'efficientnetb0')
            parts = model_name.split('_')
            
            if model_name.startswith('Baseline'):
                model_type = parts[-1].replace('efficientnetb0', 'efficientnet_b0')
                return {
                    'seg_method': 'none',
                    'input_method': 'raw',
                    'aug_type': parts[-2],
                    'model_type': model_type
                }
            else:
                seg_input = parts[0].replace('SG-', '')
                seg_parts = seg_input.split('-')
                model_type = parts[-1].replace('efficientnetb0', 'efficientnet_b0')
                
                return {
                    'seg_method': seg_parts[0].lower(),
                    'input_method': seg_parts[1].lower() if len(seg_parts) > 1 else 'raw',
                    'aug_type': parts[-2],
                    'model_type': model_type
                }
        except:
            return {'seg_method': 'none', 'input_method': 'raw', 'aug_type': 'standard', 'model_type': 'mobilenetv2'}
    
    def generate_confusion_matrix(self, y_true, y_pred, model_name):
        try:
            if len(y_true) == 0: return
            cm = confusion_matrix(y_true, y_pred)
            
            cm_path = os.path.join(Config.OUTPUT_BASE, 'confusion_matrices', f'{model_name}_cm.npy')
            np.save(cm_path, cm)
            
            plt.figure(figsize=(20, 18))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
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
            if len(test_paths) == 0: return
            
            target_layer = get_target_layer(model, config['model_type'])
            gradcam = GradCAM(model, target_layer)
            
            output_dir = os.path.join(Config.OUTPUT_BASE, 'gradcam', model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            attention_energy_scores = []
            processed_count = 0
            
            for class_idx, class_name in enumerate(self.class_names):
                class_images = [(p, l) for p, l in zip(test_paths, test_labels) if l == class_idx]
                if len(class_images) == 0: continue
                
                sampled = class_images[:Config.GRADCAM_IMAGES_PER_CLASS]
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
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = transform(img_rgb)
                        
                        with torch.no_grad():
                            output = model(img_tensor.unsqueeze(0).to(self.device))
                            pred_class = output.argmax(1).item()
                        
                        # Standard Grad-CAM
                        heatmap = gradcam.generate_heatmap(img_tensor, pred_class, use_plus_plus=False)
                        superimposed = gradcam.overlay_heatmap(img_rgb, heatmap)
                        
                        # Grad-CAM++
                        heatmap_plus = gradcam.generate_heatmap(img_tensor, pred_class, use_plus_plus=True)
                        superimposed_plus = gradcam.overlay_heatmap(img_rgb, heatmap_plus)
                        
                        # Quantitative: Calculate Attention Energy within Leaf Mask
                        # We use Otsu as a general mask to compute energy overlap
                        base_img = cv2.resize(cv2.imread(img_path), Config.IMG_SIZE)
                        base_enhanced = ImagePreprocessor.apply_clahe(base_img)
                        mask = ImagePreprocessor.segment_otsu(base_enhanced)
                        
                        energy_inside = np.sum(heatmap * (mask > 0))
                        energy_total = np.sum(heatmap)
                        if energy_total > 0:
                            attention_energy_scores.append(energy_inside / energy_total)
                        
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        axes[0].imshow(img_rgb)
                        axes[0].set_title('Original')
                        axes[0].axis('off')
                        
                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title('Grad-CAM')
                        axes[1].axis('off')
                        
                        axes[2].imshow(superimposed)
                        axes[2].set_title(f'GCAM Pred: {self.class_names[pred_class]}')
                        axes[2].axis('off')
                        
                        axes[3].imshow(superimposed_plus)
                        axes[3].set_title(f'GCAM++ Pred')
                        axes[3].axis('off')
                        
                        plt.tight_layout()
                        save_path = os.path.join(class_dir, f'gradcam_{img_idx:03d}.png')
                        plt.savefig(save_path, dpi=100, bbox_inches='tight')
                        plt.close()
                        
                        processed_count += 1
                    except:
                        continue
            
            # Save quantitative attention energy metric
            if attention_energy_scores:
                mean_energy = float(np.mean(attention_energy_scores))
                with open(os.path.join(Config.OUTPUT_BASE, 'metrics', f'{model_name}_attention_energy.json'), 'w') as f:
                    json.dump({'mean_attention_energy_in_mask': mean_energy}, f, indent=4)
                    
            print(f"  Generated {processed_count} Grad-CAM visualizations (Energy: {mean_energy:.3f})")
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
