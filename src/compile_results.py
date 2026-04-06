import os
import json
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config

class ResultsCompiler:
    """Compile results"""
    
    @staticmethod
    def create_summary_tables():
        print("\n" + "="*70)
        print("COMPILING RESULTS")
        print("="*70)
        
        try:
            results_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'all_training_results.json')
            
            if not os.path.exists(results_path):
                return
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            if len(results) == 0:
                return
            
            df = pd.DataFrame(results)
            df = df.sort_values('test_accuracy', ascending=False)
            
            csv_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'model_comparison.csv')
            df.to_csv(csv_path, index=False)
            
            print(f"\nTop 5 Models:")
            for idx, row in df.head().iterrows():
                print(f"{row['model_name']}: {row['test_accuracy']:.4f}")
                
        except Exception as e:
            print(f"Error creating summary: {e}")
    
    @staticmethod
    def create_comparison_plots():
        try:
            results_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'all_training_results.json')
            
            if not os.path.exists(results_path):
                return
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            if len(results) == 0:
                return
            
            df = pd.DataFrame(results)
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Accuracy by model type
            model_acc = df.groupby('model_type')['test_accuracy'].mean().sort_values(ascending=False)
            model_acc.plot(kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('Test Accuracy by Model Type')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy by segmentation
            seg_acc = df.groupby('seg_method')['test_accuracy'].mean().sort_values(ascending=False)
            seg_acc.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
            axes[0, 1].set_title('Test Accuracy by Segmentation')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Accuracy by augmentation
            aug_acc = df.groupby('aug_type')['test_accuracy'].mean().sort_values(ascending=False)
            aug_acc.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
            axes[1, 0].set_title('Test Accuracy by Augmentation')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Top models
            top_models = df.nlargest(min(10, len(df)), 'test_accuracy')
            axes[1, 1].barh(range(len(top_models)), top_models['test_accuracy'].values)
            axes[1, 1].set_yticks(range(len(top_models)))
            axes[1, 1].set_yticklabels(top_models['model_name'].values, fontsize=8)
            axes[1, 1].set_xlabel('Accuracy')
            axes[1, 1].set_title('Top Models')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'comparison_plots.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    @staticmethod
    def zip_all_outputs():
        try:
            print("\nCreating ZIP archive...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"results_{timestamp}.zip"
            zip_path = os.path.join(Config.OUTPUT_BASE, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(Config.OUTPUT_BASE):
                    # Skip cache and checkpoints
                    if 'cache' in root or 'checkpoints' in root:
                        continue
                    
                    if zip_filename in files:
                        files.remove(zip_filename)
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, Config.OUTPUT_BASE)
                        zipf.write(file_path, arcname)
            
            file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            print(f" ZIP: {zip_filename} ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"Error creating ZIP: {e}")
