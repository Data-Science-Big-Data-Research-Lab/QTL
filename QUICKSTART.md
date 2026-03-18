# Quick Start

This section provides a minimal path to run one training job and one experiment batch.

## 1. Environment Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare Dataset Folders

Use this structure:

```text
datasets/<dataset_name>/
  train/<class_a>/
  train/<class_b>/
  test/<class_a>/
  test/<class_b>/
```

## 3. Run One Baseline Model

```bash
python train_cc.py --dataset-file hymenoptera --classical-model resnet18 --epochs 5
```

## 4. Run One Hybrid Model

```bash
python train_cq_qiskit.py --dataset-file hymenoptera --classical-model resnet18 --n-qubits 4 --epochs 5
```

## 5. Validate All Main Pipelines

```bash
python verify_models.py
```

## 6. Launch Paper Experiments

```bash
python run_paper_experiments.py
```

## Useful Options

- `--epochs`: training epochs
- `--batch-size`: batch size
- `--learning-rate`: optimizer learning rate
- `--n-qubits`: number of qubits for hybrid models
- `--shots`: number of circuit shots when applicable
- `--dry-run`: print planned experiments without executing

## Where Results Are Saved

- Trained models: `model_saved/`
- Metrics and figures: `static/metrics/`
- Experiment logs and CSV files: output directory used by the runner
