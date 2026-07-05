import os
import time
import torch
import cv2
import numpy as np
from thop import profile
from src.models import build_optimized_model
from src.config import Config
from src.data import ImagePreprocessor

def profile_models():
    print("Profiling Computational Costs...")
    device = Config.DEVICE
    
    results = []
    
    for model_type in Config.MODEL_TYPES:
        model = build_optimized_model(model_type, 39).to(device)
        model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # FLOPs and Params using THOP
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # Model Size on disk (temporary save)
        temp_path = f"temp_{model_type}.pth"
        torch.save(model.state_dict(), temp_path)
        model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        
        # Inference Latency (GPU/CPU)
        latencies = []
        with torch.no_grad():
            for _ in range(10): # Warmup
                _ = model(dummy_input)
            for _ in range(50):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latencies.append((time.time() - start_time) * 1000) # ms
        avg_inference_latency = np.mean(latencies)
        
        results.append({
            'Model': model_type,
            'Params (M)': params / 1e6,
            'MACs (G)': macs / 1e9,
            'Size (MB)': model_size_mb,
            'Inference (ms)': avg_inference_latency
        })
        
    print("\nModel Profiling Results:")
    for res in results:
        print(f"{res['Model']}: {res['Params (M)']:.2f}M Params, {res['MACs (G)']:.2f}G MACs, {res['Size (MB)']:.2f}MB, {res['Inference (ms)']:.2f}ms")
        
    # Preprocessing Latency
    print("\nProfiling Preprocessing Latency...")
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite('temp_dummy.jpg', dummy_image)
    
    prep_methods = [
        ('Baseline (CLAHE only)', 'none', 'raw'),
        ('Otsu + Crop', 'otsu', 'cropping'),
        ('Otsu + Mask', 'otsu', 'masking'),
        ('K-Means + Crop', 'kmeans', 'cropping'),
        ('K-Means + Mask', 'kmeans', 'masking')
    ]
    
    import pandas as pd
    import json
    
    # Save Model Profiling Results
    df_models = pd.DataFrame(results)
    os.makedirs(os.path.join(Config.OUTPUT_BASE, 'final_results'), exist_ok=True)
    models_csv_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'computational_costs.csv')
    df_models.to_csv(models_csv_path, index=False)
    print(f"\nModel profiling saved to: {models_csv_path}")
    
    prep_results = {}
    
    for name, seg, inp in prep_methods:
        latencies = []
        for _ in range(50):
            start_time = time.time()
            _ = ImagePreprocessor.preprocess_image('temp_dummy.jpg', seg, inp)
            latencies.append((time.time() - start_time) * 1000)
        
        mean_latency = np.mean(latencies)
        print(f"{name}: {mean_latency:.2f} ms per image")
        prep_results[name] = float(mean_latency)
        
    os.remove('temp_dummy.jpg')
    
    # Save Preprocessing Profiling Results
    prep_json_path = os.path.join(Config.OUTPUT_BASE, 'final_results', 'preprocessing_costs.json')
    with open(prep_json_path, 'w') as f:
        json.dump(prep_results, f, indent=4)
    print(f"Preprocessing costs saved to: {prep_json_path}")

if __name__ == "__main__":
    profile_models()
