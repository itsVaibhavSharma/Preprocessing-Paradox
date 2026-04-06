import os
import json
import time
import traceback
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from src.config import Config
from src.utils import save_metadata
from src.data import create_optimized_data_loaders
from src.models import build_optimized_model


class OptimizedModelTrainer:
    """GPU-optimized trainer with mixed precision"""
    
    def __init__(self, split_data_path, n_classes, progress_tracker):
        self.split_data_path = split_data_path
        self.n_classes = n_classes
        self.progress_tracker = progress_tracker
        self.training_results = []
        self.device = Config.DEVICE
    
    def get_model_name(self, seg_method, input_method, aug_type, model_type):
        if seg_method == 'none':
            prefix = 'Baseline'
        else:
            prefix = f'SG-{seg_method.upper()}-{input_method}'
        return f"{prefix}_{aug_type}_{model_type}"
    
    def train_all_models(self):
        """Train all model combinations"""
        print("\n" + "="*70)
        print("PHASE 2: GPU-OPTIMIZED MODEL TRAINING")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Mixed Precision: {Config.MIXED_PRECISION}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Workers: {Config.NUM_WORKERS}")
        print("="*70)
        
        all_combinations = []
        
        for model_type in Config.MODEL_TYPES:
            for aug_type in Config.AUGMENTATION_TYPES:
                all_combinations.append({
                    'seg_method': 'none',
                    'input_method': 'raw',
                    'aug_type': aug_type,
                    'model_type': model_type
                })
                
                for seg_method in ['otsu', 'kmeans']:
                    for input_method in ['masking', 'cropping']:
                        all_combinations.append({
                            'seg_method': seg_method,
                            'input_method': input_method,
                            'aug_type': aug_type,
                            'model_type': model_type
                        })
        
        trained_models = self.progress_tracker.get_trained_models()
        remaining_combinations = []
        for config in all_combinations:
            model_name = self.get_model_name(
                config['seg_method'], config['input_method'],
                config['aug_type'], config['model_type']
            )
            if model_name not in trained_models:
                remaining_combinations.append(config)
        
        if len(remaining_combinations) == 0:
            print("\n All models already trained!")
            try:
                results_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'all_training_results.json')
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        self.training_results = json.load(f)
            except:
                pass
            return
        
        print(f"\nTotal: {len(all_combinations)} | Trained: {len(trained_models)} | Remaining: {len(remaining_combinations)}\n")
        
        # Load existing results
        try:
            results_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'all_training_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.training_results = json.load(f)
        except:
            pass
        
        for idx, config in enumerate(remaining_combinations, 1):
            print(f"\n{'='*70}")
            print(f"Training Model {len(trained_models) + idx}/{len(all_combinations)}")
            print(f"{'='*70}")
            
            model_name = self.get_model_name(
                config['seg_method'], config['input_method'],
                config['aug_type'], config['model_type']
            )
            
            print(f"Configuration: {model_name}")
            
            try:
                result = self.train_single_model(
                    model_name=model_name,
                    **config
                )
                
                if result:
                    self.training_results.append(result)
                    save_metadata(self.training_results, 'all_training_results.json')
                
            except Exception as e:
                print(f"\n Error training {model_name}: {e}")
                traceback.print_exc()
                continue
        
        save_metadata(self.training_results, 'all_training_results.json')
        print(f"\n Training complete!")
    
    def train_single_model(self, model_name, seg_method, input_method, aug_type, model_type):
        """Train single model with mixed precision and epoch resume"""
        
        # Check if already trained
        if self.progress_tracker.is_model_trained(model_name):
            print(f"Model already trained, loading results...")
            return {}
        
        start_time = time.time()
        
        try:
            # Create optimized data loaders
            train_loader, val_loader, test_loader, test_data = create_optimized_data_loaders(
                self.split_data_path, seg_method, input_method, aug_type
            )
            
            if len(train_loader) == 0:
                raise ValueError("Empty data loader!")
            
            # Build optimized model
            model = build_optimized_model(model_type, self.n_classes)
            model = model.to(self.device)
            
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Parameters: {n_params:,}")
            
            # Loss and optimizer (AdamW for better results)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                                   weight_decay=Config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
            
            # Mixed precision scaler
            scaler = GradScaler() if Config.MIXED_PRECISION else None
            
            # Training history
            history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch = 0
            start_epoch = 0
            
            model_path = os.path.join(Config.OUTPUT_BASE, 'models', f'{model_name}.pth')
            checkpoint_path = os.path.join(Config.OUTPUT_BASE, 'checkpoints', f'{model_name}_checkpoint.pth')
            
            # Try to resume from checkpoint
            training_state = self.progress_tracker.get_training_state(model_name)
            if training_state and os.path.exists(checkpoint_path):
                print(f"  Resuming from epoch {training_state['last_epoch']}...")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scaler and 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                history = checkpoint['history']
                best_val_loss = training_state['best_val_loss']
                start_epoch = training_state['last_epoch']
            
            # Training loop
            for epoch in range(start_epoch, Config.EPOCHS):
                try:
                    # Train
                    model.train()
                    train_loss = 0
                    train_correct = 0
                    train_total = 0
                    
                    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS}', leave=False)
                    for batch_idx, (images, labels) in enumerate(pbar):
                        try:
                            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                            
                            optimizer.zero_grad()
                            
                            # Mixed precision forward pass
                            if Config.MIXED_PRECISION:
                                with autocast():
                                    outputs = model(images)
                                    loss = criterion(outputs, labels)
                                
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                            
                            # Update metrics
                            train_loss += loss.item() * images.size(0)
                            _, predicted = outputs.max(1)
                            train_total += labels.size(0)
                            train_correct += predicted.eq(labels).sum().item()
                            
                            # Update progress bar less frequently
                            if batch_idx % 10 == 0:
                                pbar.set_postfix({
                                    'loss': f'{loss.item():.4f}',
                                    'acc': f'{100.*train_correct/train_total:.1f}%'
                                })
                        except Exception as e:
                            print(f"Batch error: {e}")
                            continue
                    
                    if train_total == 0:
                        continue
                    
                    train_loss /= train_total
                    train_acc = train_correct / train_total
                    
                    # Validate
                    model.eval()
                    val_loss = 0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for images, labels in val_loader:
                            try:
                                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                                
                                if Config.MIXED_PRECISION:
                                    with autocast():
                                        outputs = model(images)
                                        loss = criterion(outputs, labels)
                                else:
                                    outputs = model(images)
                                    loss = criterion(outputs, labels)
                                
                                val_loss += loss.item() * images.size(0)
                                _, predicted = outputs.max(1)
                                val_total += labels.size(0)
                                val_correct += predicted.eq(labels).sum().item()
                            except:
                                continue
                    
                    if val_total == 0:
                        continue
                    
                    val_loss /= val_total
                    val_acc = val_correct / val_total
                    
                    # Update history
                    history['train_loss'].append(train_loss)
                    history['train_accuracy'].append(train_acc)
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_acc)
                    
                    print(f'Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
                    
                    # LR scheduling
                    scheduler.step()
                    
                    # Save checkpoint
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history,
                        'best_val_loss': best_val_loss
                    }
                    if scaler:
                        checkpoint['scaler_state_dict'] = scaler.state_dict()
                    
                    torch.save(checkpoint, checkpoint_path)
                    self.progress_tracker.save_training_state(model_name, epoch + 1, best_val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        patience_counter = 0
                        torch.save(model.state_dict(), model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= Config.EARLY_STOP_PATIENCE:
                            print(f'Early stopping at epoch {epoch+1}')
                            break
                
                except Exception as e:
                    print(f"Epoch error: {e}")
                    continue
            
            training_time = time.time() - start_time
            
            # Load best model
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
            
            # Evaluate
            print(f"Evaluating on test set...")
            model.eval()
            test_correct = 0
            test_total = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    try:
                        images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                        
                        if Config.MIXED_PRECISION:
                            with autocast():
                                outputs = model(images)
                        else:
                            outputs = model(images)
                        
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()
                        
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
                    except:
                        continue
            
            test_acc = test_correct / test_total if test_total > 0 else 0
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Save history
            history_path = os.path.join(Config.OUTPUT_BASE, 'training_history', f'{model_name}_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Plot curves
            self.plot_training_curves(model_name, history)
            
            result = {
                'model_name': model_name,
                'seg_method': seg_method,
                'input_method': input_method,
                'aug_type': aug_type,
                'model_type': model_type,
                'parameters': int(n_params),
                'training_time_seconds': float(training_time),
                'best_epoch': int(best_epoch),
                'best_val_loss': float(best_val_loss),
                'best_val_accuracy': float(max(history['val_accuracy']) if history['val_accuracy'] else 0),
                'test_accuracy': float(test_acc),
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1_score': float(f1)
            }
            
            print(f"\n{'='*50}")
            print(f"Results: acc={test_acc:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")
            print(f"Time: {training_time/60:.1f} min")
            print(f"{'='*50}")
            
            # Mark as trained
            self.progress_tracker.mark_model_trained(model_name)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            traceback.print_exc()
            raise
    
    def plot_training_curves(self, model_name, history):
        """Plot training curves"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            epochs = range(1, len(history['train_accuracy']) + 1)
            
            axes[0].plot(epochs, history['train_accuracy'], label='Train')
            axes[0].plot(epochs, history['val_accuracy'], label='Val')
            axes[0].set_title(f'{model_name} - Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(epochs, history['train_loss'], label='Train')
            axes[1].plot(epochs, history['val_loss'], label='Val')
            axes[1].set_title(f'{model_name} - Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            save_path = os.path.join(Config.OUTPUT_BASE, 'training_curves', f'{model_name}_curves.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except:
            pass
