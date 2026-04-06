import os
import sys
import time
import traceback
import torch
import warnings

warnings.filterwarnings('ignore')

from src.config import Config
from src.utils import ProgressTracker, create_directory_structure, get_class_names
from src.data import DatasetSplitter, PreprocessingSampleSaver
from src.train import OptimizedModelTrainer
from src.evaluate import ModelEvaluator
from src.compile_results import ResultsCompiler

def main():
    """GPU-optimized main pipeline"""
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTION - GPU OPTIMIZED (RTX 5090)")
    print("Updated with SqueezeNet & Fixed Augmentation Samples")
    print("="*70)
    
    start_time = time.time()
    
    progress_tracker = ProgressTracker()
    create_directory_structure()
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Mixed Precision: {Config.MIXED_PRECISION}")
        print(f"   Batch Size: {Config.BATCH_SIZE}")
        print(f"   Workers: {Config.NUM_WORKERS}")
        print(f"   Cache: {'Enabled' if Config.USE_CACHE else 'Disabled'}")
    
    class_names = get_class_names(Config.DATASET_PATH)
    n_classes = len(class_names)
    
    if n_classes == 0:
        print("ERROR: No classes found!")
        return
    
    print(f"\nDataset: {n_classes} classes")
    
    try:
        # Phase 0: Split
        splitter = DatasetSplitter(
            Config.DATASET_PATH,
            os.path.join(Config.OUTPUT_BASE, 'split_data'),
            progress_tracker
        )
        split_info = splitter.split_and_copy()
        
        # Phase 1: Save preprocessing samples
        sample_saver = PreprocessingSampleSaver(
            os.path.join(Config.OUTPUT_BASE, 'split_data'),
            progress_tracker
        )
        sample_saver.save_preprocessing_samples()
        
        # Phase 2: Train
        trainer = OptimizedModelTrainer(
            os.path.join(Config.OUTPUT_BASE, 'split_data'),
            n_classes,
            progress_tracker
        )
        trainer.train_all_models()
        
        # Phase 3: Evaluate
        evaluator = ModelEvaluator(
            os.path.join(Config.OUTPUT_BASE, 'split_data'),
            class_names,
            progress_tracker
        )
        evaluator.evaluate_all_models()
        
        # Compile results
        ResultsCompiler.create_summary_tables()
        ResultsCompiler.create_comparison_plots()
        ResultsCompiler.zip_all_outputs()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nTotal time: {total_time/3600:.2f} hours")
        print(f"Results: {os.path.abspath(Config.OUTPUT_BASE)}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n Error: {e}")
        traceback.print_exc()
        print("\nRe-run to resume from checkpoint!")

if __name__ == "__main__":
    if not os.path.exists(Config.DATASET_PATH):
        print(f"ERROR: Dataset not found: {Config.DATASET_PATH}")
        sys.exit(1)
    
    if torch.cuda.is_available():
        print(f"\n GPU Ready: {torch.cuda.get_device_name(0)}")
    else:
        print("\n No GPU - will be slow!")
    
    main()
