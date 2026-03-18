# Quantum Transfer Learning (QTL)

This repository contains the official codebase used in our paper on hybrid classical-quantum transfer learning for image classification.

The project provides:
- Classical transfer learning baselines.
- Hybrid Qiskit models (ideal and noisy simulation).
- Hybrid PennyLane models (ideal and noisy simulation).
- Reproducible experiment runners for paper-level evaluation.
- Optional execution of trained Qiskit models on IBM Quantum hardware.

## Scope

The code is designed for controlled comparative experiments across:
- Multiple datasets.
- Multiple CNN backbones.
- Multiple training approaches (classical and hybrid quantum).

Outputs include trained models, logs, CSV summaries, and evaluation plots.

## Repository Layout

- `train_cc.py`: classical transfer learning training.
- `train_cq_qiskit.py`: hybrid Qiskit training (ideal/clean simulation).
- `train_cq_qiskit_noisy.py`: hybrid Qiskit training with realistic noise models.
- `train_cq_pennylane.py`: hybrid PennyLane training (ideal/clean simulation).
- `train_cq_pennylane_noisy.py`: hybrid PennyLane training with realistic noise.
- `run_paper_experiments.py`: systematic paper experiment runner.
- `run_complete_benchmark.py`: large benchmark runner across configurations.
- `verify_models.py`: quick validation before full experiments.
- `test_qiskit_real_hardware.py`: inference/evaluation on IBM Quantum hardware.
- `datasets/`: expected dataset directory structure.
- `static/metrics/`: generated figures and metric summaries.
- `model_saved/`: trained checkpoints.

## Environment Requirements

- Python 3.8+
- PyTorch + torchvision
- Qiskit
- PennyLane
- scikit-learn
- matplotlib

See `requirements.txt` for the complete dependency set.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Detailed instructions are available in `INSTALLATION.md`.

## Dataset Preparation

Expected directory format:

```text
datasets/
  hymenoptera/
    train/<class_name>/
    test/<class_name>/
  brain_tumor/
    train/<class_name>/
    test/<class_name>/
  cats_dogs/
    train/<class_name>/
    test/<class_name>/
  solar_dust/
    train/<class_name>/
    test/<class_name>/
```

See `datasets/README.md` for exact conventions and validation notes.

## Training Commands

### Classical baseline

```bash
python train_cc.py --dataset-file hymenoptera --classical-model resnet18 --epochs 10
```

### Hybrid Qiskit (ideal)

```bash
python train_cq_qiskit.py --dataset-file hymenoptera --classical-model resnet18 --n-qubits 4 --epochs 10
```

### Hybrid Qiskit (noisy)

```bash
python train_cq_qiskit_noisy.py --dataset-file hymenoptera --classical-model resnet18 --n-qubits 4 --epochs 10
```

### Hybrid PennyLane (ideal)

```bash
python train_cq_pennylane.py --dataset-file hymenoptera --classical-model resnet18 --n-qubits 4 --epochs 10
```

### Hybrid PennyLane (noisy)

```bash
python train_cq_pennylane_noisy.py --dataset-file hymenoptera --classical-model resnet18 --n-qubits 4 --epochs 10
```

## Reproducible Experiment Runners

1. Quick sanity check before long runs:

```bash
python verify_models.py
```

2. Paper-level systematic experiments:

```bash
python run_paper_experiments.py
```

3. Full benchmark matrix:

```bash
python run_complete_benchmark.py
```

Additional flags are documented in `EXPERIMENTS_README.md` and `QUICKSTART.md`.

## IBM Quantum Hardware Testing

After training a compatible Qiskit hybrid model, you can evaluate it on real hardware:

```bash
python test_qiskit_real_hardware.py --model-path model_saved/<model_file>.pth --backend ibm_nairobi
```

Use `--estimate-only` to inspect runtime cost before submitting hardware jobs.

## Outputs

Typical artifacts:
- Checkpoints in `model_saved/`.
- Plots in `static/metrics/`.
- Experiment summaries in CSV/TXT files under output directories.

## Reproducibility Notes

- Set fixed random seeds when comparing approaches.
- Keep dataset splits unchanged across methods.
- Use the same backbone and training budget for fair comparisons.

## Citation

If you use this repository, cite the corresponding paper when available.

## License

This project is released under the MIT License.
