# The Preprocessing Paradox: Architecture-Aware Segmentation for Plant Disease Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19439063.svg)](https://doi.org/10.5281/zenodo.19439063)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Authors:** Vaibhav Sharma, Rajni Ranjan Singh Makwana  
**Affiliation:** Centre for Artificial Intelligence, Madhav Institute of Technology and Science, Gwalior, India  
**Paper:** *"Preprocessing Paradox: Enhancing Lightweight CNNs for Plant Disease Classification via Explainable AI"*  
**Journal:** [The Visual Computer](https://link.springer.com/journal/371) — Under Review

---

## Overview

This repository contains the **complete, reproducible implementation** for our systematic study of how classical image segmentation preprocessing interacts with CNN architecture capacity across plant disease classification tasks.

We reveal a robust, statistically validated **preprocessing paradox**: the same segmentation operation that *improves* a low-capacity model can *harm* a high-capacity model — and the boundary between these behaviors can be precisely located along the architecture parameter spectrum.

### Key Finding

> Otsu-based cropping improves SqueezeNet (0.74M) from **96.87% ± 1.07%** → **97.50% ± 0.75%**, while EfficientNet-B0 (4.06M) achieves its best accuracy of **99.61% ± 0.21%** *without* any segmentation (*p* = 0.118, NS). K-Means masking with heavy augmentation collapses SqueezeNet to **68.93% ± 5.52%** — a catastrophic 28 pp drop validated by McNemar's test (χ² = 303.08, *p* = 7.03 × 10⁻⁶⁸).

---

## Study Design

| Dimension | Options |
|---|---|
| **Architectures** | SqueezeNet 1.1, ShuffleNetV2-1.0×, MobileNetV3-Small, MobileNetV2, EfficientNet-B0 |
| **Segmentation** | None (Baseline/CLAHE only), Otsu thresholding, K-Means clustering (HSV, 3 clusters) |
| **Input preparation** | Cropping (bounding box), Masking (element-wise) |
| **Augmentation** | Standard (±20° rotation, flip, ±10% translate/zoom), Heavy (±40°, ±20%, brightness, shear) |
| **Total configurations** | **50** |
| **Random seeds** | 5 per configuration (seeds 42–46) |
| **Total trained models** | **250** |
| **Dataset** | PlantVillage (54,305 images, 39 classes, 14 crops) |

---

## Architecture Capacity Spectrum

| Model | Params | MACs | Size | Inference |
|---|---|---|---|---|
| SqueezeNet 1.1 | 0.742M | 0.266G | 2.86 MB | 1.38 ms |
| ShuffleNetV2-1.0× | 1.294M | 0.152G | 5.11 MB | 3.60 ms |
| MobileNetV3-Small | 1.558M | 0.061G | 6.12 MB | 2.90 ms |
| MobileNetV2 | 2.900M | 0.327G | 11.33 MB | 3.05 ms |
| EfficientNet-B0 | 4.058M | 0.414G | 15.87 MB | 4.61 ms |

*Hardware-profiled on NVIDIA RTX 5090 GPU (50 runs after 10 warm-up).*

---

## Repository Structure

```
Github_Code/
├── main.py                      # Entry point — runs full pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Core source modules
│   ├── __init__.py
│   ├── config.py                # Central Config class (paths, hyperparams, seeds)
│   ├── data.py                  # Dataset, DataLoaders, preprocessing (CLAHE/Otsu/K-Means)
│   ├── models.py                # Model initialization with pretrained weights + custom heads
│   ├── train.py                 # OptimizedModelTrainer (AdamW, cosine annealing, AMP)
│   ├── evaluate.py              # ModelEvaluator (accuracy, F1, McNemar, Grad-CAM, AE metric)
│   ├── compile_results.py       # ResultsCompiler (CSV summaries, plots)
│   ├── utils.py                 # ProgressTracker, helpers
│   └── profile_models.py       # Hardware profiling (params, MACs, latency)
│
└── results/                     # All experimental outputs
    ├── final_results/           # Summary CSVs and JSON aggregates
    │   ├── all_training_results.json    # Complete 250-model raw results
    │   ├── model_comparison.csv         # Per-configuration mean/std accuracy & F1
    │   ├── computational_costs.csv      # Deployment metrics per architecture
    │   ├── preprocessing_costs.json     # Preprocessing latency measurements
    │   └── dataset_split_info.json      # Stratified split statistics
    ├── training_curves/         # Loss/accuracy plots per configuration × seed
    ├── training_history/        # Epoch-by-epoch JSON histories (250 files)
    ├── confusion_matrices/      # Per-configuration normalized confusion matrices
    ├── metrics/                 # Per-class precision/recall/F1 classification reports
    ├── gradcam/                 # Grad-CAM overlay images (key configurations)
    │   ├── baseine standard squeezenet.png
    │   ├── otsucrop standard squeezenet.png
    │   ├── baseine efficientnet standard.png
    │   └── otsucrop standard efficient.png
    └── progress.json            # Training progress tracker
```

---

## Setup

### Requirements

Python 3.8+ with CUDA-capable GPU recommended (trained on NVIDIA RTX 5090, 32 GB VRAM).

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
```
torch
torchvision
opencv-python
numpy
pandas
matplotlib
seaborn
tqdm
scikit-learn
statsmodels>=0.13.0
thop>=0.1.1.post2209072238
```

### Dataset

Download the **PlantVillage** dataset (without augmentation split):
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)
- Or via `tensorflow_datasets`: `tfds.load("plant_village")`

Set the dataset path:

```bash
# Linux / macOS
export DATASET_PATH="/path/to/Plant_leaf_diseases_dataset_without_augmentation"

# Windows CMD
set DATASET_PATH=C:\path\to\Plant_leaf_diseases_dataset_without_augmentation

# Windows PowerShell
$env:DATASET_PATH="C:\path\to\Plant_leaf_diseases_dataset_without_augmentation"
```

Or modify `DATASET_PATH` directly in [`src/config.py`](src/config.py).

---

## Running the Pipeline

### Full Experiment (all 50 configurations × 5 seeds)

```bash
python main.py
```

This runs the complete pipeline:
1. **Data splitting** — stratified 80/10/10 train/val/test split
2. **Preprocessing** — CLAHE → optional Otsu or K-Means → crop or mask
3. **Training** — AdamW + cosine annealing, 50 epochs, AMP, early stopping (patience=8)
4. **Evaluation** — accuracy, macro-F1, McNemar's test, Grad-CAM + AE metric
5. **Compilation** — summary CSVs, comparison plots

### Profile Hardware Costs

```bash
python -c "from src.profile_models import profile_all; profile_all()"
```

### Reproduce Specific Configuration

```python
from src.config import Config
from src.train import OptimizedModelTrainer

cfg = Config(
    model_name="squeezenet",        # squeezenet | shufflenetv2 | mobilenetv3 | mobilenetv2 | efficientnet_b0
    seg_method="otsu",              # none | otsu | kmeans
    input_type="cropping",          # cropping | masking
    aug_type="standard",            # standard | heavy
    seed=42
)
trainer = OptimizedModelTrainer(cfg)
trainer.train()
```

---

## Key Results Summary

### Top Configurations by Accuracy (Mean ± Std, 5 seeds)

| Rank | Configuration | Accuracy | F1 |
|---|---|---|---|
| 1 | EfficientNet-B0 — Baseline, Std. | **99.61% ± 0.21%** | 0.9961 |
| 2 | EfficientNet-B0 — Baseline, Heavy | 99.48% ± 0.31% | 0.9948 |
| 3 | ShuffleNetV2 — Baseline, Std. | 99.35% ± 0.12% | 0.9935 |
| 28 | **SqueezeNet — Otsu+Crop, Std.** *(best for SqueezeNet)* | **97.50% ± 0.75%** | 0.9749 |
| 31 | SqueezeNet — Baseline, Std. | 96.87% ± 1.07% | 0.9687 |
| 49 | SqueezeNet — K-Means+Mask, Heavy | **68.93% ± 5.52%** | 0.6793 |
| 50 | SqueezeNet — Otsu+Mask, Heavy | 68.56% ± 12.76% | 0.6656 |

### Preprocessing Overhead

| Method | Latency | vs. Baseline |
|---|---|---|
| Baseline (CLAHE only) | 2.16 ms/image | — |
| Otsu + Crop | **1.01 ms/image** | −53% |
| Otsu + Mask | 0.94 ms/image | −56% |
| K-Means + Crop | 42.24 ms/image | **+1857%** |
| K-Means + Mask | 42.65 ms/image | +1875% |

### Attention Energy (Grad-CAM AE metric)

| Configuration | AE | Interpretation |
|---|---|---|
| SqueezeNet — Baseline | 0.4194 | Diffuse, scattered attention |
| SqueezeNet — Otsu+Crop | **0.4463** | +6.4%: attention redirected to lesions ✓ |
| SqueezeNet — K-Means+Mask | 0.3977 | *Below baseline* — masks misdirect attention ✗ |
| EfficientNet-B0 — Baseline | 0.3511 | Stable, disease-specific without external guidance |
| EfficientNet-B0 — Otsu+Crop | 0.3521 | Near-identical — segmentation redundant ✓ |

---

## Architecture-Aware Preprocessing Framework

Based on the 50-configuration study:

| Model Capacity | Recommendation |
|---|---|
| **> 2M params** (MobileNetV2, EfficientNet-B0) | ✅ Use Baseline (CLAHE only). K-Means contraindicated. |
| **1.3M–2M params** (ShuffleNetV2, MobileNetV3) | ⚠️ Evaluate Otsu+Crop (Std. aug). Avoid masking and K-Means. |
| **< 1M params** (SqueezeNet) | ✅ Apply Otsu+Crop (Std. aug). K-Means masking → catastrophic collapse risk. |
| **Universal** | 🔁 Always validate with ≥5 seeds before finalizing preprocessing decisions. |

---

## Reproducing Figures from the Paper

| Figure | Script / Data |
|---|---|
| Fig 2 — Architecture diagram | Generated image (see paper) |
| Fig 3 — Preprocessing pipeline | `src/data.py` outputs to `outputs/preprocessed_samples/` |
| Fig 4 — Training curves | `results/training_curves/` |
| Fig 5 — Grad-CAM composite | `results/gradcam/` |
| Fig 6 — 50-config summary plots | `results/final_results/comparison_plots.png` |
| Fig 7 — Decision flowchart | Generated image (see paper) |
| Table 1 — All 50 configs | `results/final_results/model_comparison.csv` |
| Table 3 — Deployment costs | `results/final_results/computational_costs.csv` |
| Table 4 — McNemar's tests | `src/evaluate.py` → `mcnemar_test()` |
| Table 6 — Attention Energy | `src/evaluate.py` → `compute_attention_energy()` |

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{sharma2026preprocessing,
  title   = {Preprocessing Paradox: Enhancing Lightweight CNNs for Plant Disease
             Classification via Explainable AI},
  author  = {Sharma, Vaibhav and Makwana, Rajni Ranjan Singh},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Under Review},
  doi     = {10.5281/zenodo.19439063}
}
```

**Zenodo permanent archive:** https://doi.org/10.5281/zenodo.19439063

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

**Vaibhav Sharma**  
Centre for Artificial Intelligence  
Madhav Institute of Technology and Science, Gwalior, India  
GitHub: [@itsVaibhavSharma](https://github.com/itsVaibhavSharma)
