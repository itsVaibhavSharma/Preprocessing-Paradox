# Preprocessing Paradox: Enhancing Lightweight CNNs for Plant Disease Classification via Explainable AI

[![DOI](https://zenodo.org/badge/1202775468.svg)](https://doi.org/10.5281/zenodo.19439063)

**Authors**: Vaibhav Sharma, Rajni Ranjan Singh Makwana  
**Affiliation**: Centre for Artificial Intelligence, Madhav Institute of Technology and Science, Gwalior, India

This repository contains the official implementation for the paper **"Preprocessing Paradox: Enhancing Lightweight CNNs for Plant Disease Classification via Explainable AI"**, submitted to *[The Visual Computer](https://link.springer.com/journal/371)*.

## Quick Summary

This study investigates how common image segmentation techniques (like Otsu thresholding and K-Means clustering) differently impact lightweight models (SqueezeNet) versus high-capacity models (MobileNetV2) in plant disease classification. 

Our findings reveal a **"preprocessing paradox"**: while segmentation dramatically improves accuracy for lightweight models by helping them focus on the diseased regions, it actually degrades the performance of high-capacity models, which already develop robust implicit attention mechanisms. We provide empirical results and Gradient-weighted Class Activation Mapping (Grad-CAM) visual explanations to support these findings, offering practical guidelines for researchers deploying models in resource-constrained agricultural settings.

## Experimental Setup & Key Results

We explore 20 experimental configurations by varying:
- **Model Architectures:** SqueezeNet (0.74M parameters) vs. MobileNetV2 (2.9M parameters).
- **Segmentation Methods:** None (Baseline), Otsu Thresholding, K-Means Clustering.
- **Input Preparation:** Raw, Masking, Cropping.
- **Augmentation Profiles:** Standard vs. Heavy augmentation.

### Key Results
- **MobileNetV2 (Heavy Capacity):** Achieved **99.78%** accuracy on raw, unsegmented images. Applying Otsu segmentation reduced its accuracy slightly to 99.53%, proving high-capacity models perform optimally without external segmentation constraints.
- **SqueezeNet (Lightweight):** Achieved 97.42% accuracy on raw images. Applying **Otsu-based cropping** boosted it to **98.35%**, demonstrating how segmentation guides smaller parameter networks.
- **Otsu vs. K-Means:** Otsu consistently outperformed K-Means across all subsets, yielding smoother, more robust leaf isolation.
- **Explainability:** Grad-CAM heatmaps demonstrate that unsegmented MobileNetV2 natively focuses on diseased spots. Unsegmented SqueezeNet diffuses attention across noisy backgrounds, but Otsu segmentation forcefully constraints this attention precisely onto the leaf, leading to enhanced performance.

## Dataset Details

We utilize the **PlantVillage dataset** for our experiments.

- **Dataset Link**: [TensorFlow Datasets: PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village)
- Download the dataset (the `Plant_leaf_diseases_dataset_without_augmentation` version).
- You can configure the `DATASET_PATH` environment variable, or modify it directly in `src/config.py`.

## Relevant Manuscript & Citation

**Important**: The open-source code hosted here is directly related to the manuscript submitted to *[The Visual Computer](https://link.springer.com/journal/371)*. We strongly urge readers and researchers who find this work useful or replicate these experiments to cite our manuscript:

```bibtex
@article{sharma2026preprocessing,
  title={Preprocessing Paradox: Enhancing Lightweight CNNs for Plant Disease Classification via Explainable AI},
  author={Sharma, Vaibhav and Makwana, Rajni Ranjan Singh},
  journal={The Visual Computer},
  year={2026},
  note={Under Review}
}
```

## Setup & Usage Guidelines

### 1. Requirements and Dependencies
To ensure reproducibility, install the necessary dependencies via Python 3.8+ (preferably via Anaconda/Miniconda or a `venv`):

```bash
pip install -r requirements.txt
```

### 2. Environment Variables
```bash
# Set environment variable pointing to the dataset
# Linux/Mac
export DATASET_PATH="/path/to/Plant_leaf_diseases_dataset_without_augmentation"
# Windows CMD
set DATASET_PATH=C:\path\to\Plant_leaf_diseases_dataset_without_augmentation
# Windows PowerShell
$env:DATASET_PATH="C:\path\to\Plant_leaf_diseases_dataset_without_augmentation"
```

### 3. Running the Pipeline
To retrain models, evaluate performance, and produce heatmaps representing the "preprocessing paradox", merely run the main script:

```bash
python main.py
```

### 4. Output Directories

Running the pipeline will automatically generate a structured `outputs/` directory to store all experimental findings cleanly. Below is the generated schema and what it holds:

```text
outputs/
├── checkpoints/             # Intermediate model state dicts
├── confusion_matrices/      # .npy arrays and .png heatmap visualizations
├── final_results/           # CSV summary files and metric JSONs 
├── gradcam/                 # Generated Grad-CAM overlay plot images classes
├── metrics/                 # Detailed classification reports per experiment
├── models/                  # Final .pth trained model weights
├── preprocessed_cache/      # Cached segmentations and resized arrays for speed
├── preprocessed_samples/    # Debug image representations of CLAHE, Otsu, K-Means etc.
├── split_data/              # Train/Val/Test subsets of images 
├── training_curves/         # Plot graphs for loss and accuracy
└── training_history/        # JSON histories storing epoch-by-epoch loss/accuracy values
```

### 5. Repository Structure & Modules

The repository employs a scalable modular structure:
- `main.py`: The root script managing the full pipeline execution (Data Splitting -> Preprocessing -> Training -> Evaluation -> Compilation).
- `src/config.py`: Exposes the central `Config` class, holding paths, hyper-parameters (e.g., learning rate, epochs), and augmentation intensities.
- `src/utils.py`: Contains helper functions such as `ProgressTracker` (allowing for training resumption if interrupted).
- `src/data.py`: Centralizes PyTorch `Dataset` structures and customized DataLoaders. It performs the dynamic CLAHE, Otsu, and K-Means segmentation along with masking/cropping operations and caching.
- `src/models.py`: Handles model initialization. Pre-trained weights are pulled while defining updated classifier stages tailored to MobileNetV2 and SqueezeNet.
- `src/train.py`: Contains the `OptimizedModelTrainer` responsible for mixed-precision training loops and learning rate scheduling (Cosine Annealing).
- `src/evaluate.py`: Houses the `ModelEvaluator` evaluating test benchmarks and handles the generation of Grad-CAM heatmaps overlaying raw images with their learned attention.
- `src/compile_results.py`: Implements `ResultsCompiler` to automatically stitch metrics into summary CSVs or analytical plots.

## Accessibility and Long-Term Usability
To ensure transparency and replication simplicity, data and artifacts (once fully published) will carry a formal Digital Object Identifier (DOI), bolstering long-term accessibility and usability within the scientific and agriculture-AI community.
