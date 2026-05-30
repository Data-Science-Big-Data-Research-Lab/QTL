# Quantum Transfer Learning (QTL) Benchmarking Suite

Welcome to the **Quantum Transfer Learning (QTL) Benchmarking Framework** repository! 

This codebase is a comprehensive suite designed to train, evaluate, and benchmark hybrid classical-quantum models against pure classical counterparts. It supports combining deep classical backbones (such as ResNet-18, MobileNet-V2, EfficientNet-B0, and RegNet-X-400MF) with both classical classifiers and Variational Quantum Circuits (VQCs) simulated using **PennyLane** and **Qiskit** (supporting both ideal and noisy environments modeled on real hardware).

---

## Key Features

* **Fair Quantum-Classical Benchmarks:** Evaluates quantum models against parameter-matched Multi-Layer Perceptron (MLP) baselines to ensure comparison is scientifically valid.
* **Unified Simulator Frontends:** Seamlessly switch between PennyLane and Qiskit simulator architectures.
* **Realistic Noise Calibration:** Implements discrete and composite quantum noise channels calibrated against IBM Quantum Hardware (IBM Heron r2).
* **Hardware & Energy Benchmarking:** Measures training/validation execution time and logs electrical carbon footprints (kWh) via **CodeCarbon**.
* **Rich Visualizations:** Automatically logs training metrics (Loss, Accuracy, Precision, Recall, F1, AUC-ROC) and generates confusion matrices, learning curves, ROC-AUC curves, and Precision-Recall plots for every experiment.
* **Statistical Rigor:** Easy multi-seed runs with scripts to automatically aggregate results and run pairwise t-tests/Wilcoxon tests.

---

## Project Structure

```
├── data/
│   ├── download_datasets.py    # Auto-downloads public datasets and guides Kaggle imports
│   ├── loader.py               # Image dataset loading & preprocessing pipelines
│   └── tabular_loader.py       # Pipelines for tabular dataset structures
├── heads/
│   ├── classical_head.py       # Classical Linear and MLP classifier heads
│   ├── pennylane_head.py       # PennyLane VQC classifier head (ideal & noisy)
│   └── qiskit_head.py          # Qiskit VQC classifier head (ideal & noisy)
├── paper/
│   ├── paper_sections.tex      # LaTeX source code of the paper
│   └── tables/                 # Automatically generated LaTeX results tables
├── results/                    # CSV performance metrics and automatically generated plots
├── runner.py                   # Main experiment orchestrator script (sweeps & filters)
├── manager.py                  # Friendly interactive TUI menu to control experiments
├── trainer.py                  # Core PyTorch training loop with checkpoint-resume logic
├── requirements.txt            # Python package dependencies
└── config.yaml                 # Master configuration file for runs
```
*Note: PyTorch model weights (`.pt`/`.pth` checkpoints) have been excluded to keep the repository light.*

---

## Running on a Normal PC

You do not need a supercomputer or a real quantum computer to run this project! You can run it locally on any desktop or laptop using standard python CPU or GPU acceleration.

### 1. Prerequisites & Installation
Ensure you have Python 3.8+ installed (Python 3.10 or 3.12 is recommended). 

Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Datasets
We use the **Hymenoptera** dataset as a lightweight starting point. You can automatically download and configure it with:
```bash
python data/download_datasets.py --dataset hymenoptera
```
For other datasets (Brain Tumor MRI, Cats vs Dogs, Solar Panel Dust), run the helper script to view download links and structure requirements:
```bash
python data/download_datasets.py --list
```

### 3. Execution Options

#### Option A: The Interactive Menu (Recommended)
We provide an interactive Text User Interface (TUI) menu to control execution, view completed runs, and print LaTeX tables. Simply run:
```bash
python manager.py
```

#### Option B: Direct Command Line (CLI)
You can launch custom training runs using the CLI orchestrator `runner.py`.

* **Classical Baseline (ResNet-18 + Linear Head):**
  ```bash
  python runner.py --config config.yaml --dataset hymenoptera --backbone resnet18 --head linear
  ```

* **Quantum PennyLane (ResNet-18 + PennyLane Ideal VQC):**
  ```bash
  python runner.py --config config.yaml --dataset hymenoptera --backbone resnet18 --head pl_ideal
  ```

* **Quantum Qiskit (MobileNet-V2 + Qiskit Noisy VQC):**
  ```bash
  python runner.py --config config.yaml --dataset hymenoptera --backbone mobilenetv2 --head qk_noisy
  ```

* **Parallel Execution (e.g., using 4 CPU workers):**
  ```bash
  python runner.py --config config.yaml --dataset hymenoptera --parallel 4
  ```

---

## Evaluation & Visualization

Once training finishes, results are stored inside the `results/` folder under a dedicated run directory (e.g. `results/001_pc1_hymenoptera_resnet18_pl_ideal/`).
This directory will contain:
* `runs.csv`: Overall run configuration and summary metrics.
* `training_log.csv`: Training and validation loss/accuracy recorded per epoch.
* `predictions.csv`: Model predictions, ground truth labels, and raw output probabilities for the test set.
* `plots/`: Automatically generated PNG plots, including:
  * **Learning Curves:** Train/validation loss and accuracy.
  * **Confusion Matrix:** Detail of predictions across classes.
  * **ROC and Precision-Recall Curves:** Multi-threshold performance visualizers.
  * **Summary Plot:** A combined overview grid containing all curves.

To aggregate all completed run statistics and generate LaTeX tables for a manuscript, run:
```bash
python generate_tables.py
```
This will automatically parse the `results/` directory, compute statistical significances, and output the tables under the `paper/tables/` folder.
