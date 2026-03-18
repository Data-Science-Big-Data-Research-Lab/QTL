# Installation Guide

This guide describes a clean setup for running all QTL experiments.

## 1. Prerequisites

- Python 3.8 or newer
- pip
- Git
- Optional: CUDA-enabled GPU for faster training

## 2. Clone Repository

```bash
git clone https://github.com/Data-Science-Big-Data-Research-Lab/QTL.git
cd QTL
```

## 3. Create and Activate Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Verify Installation

```bash
python -c "import torch, qiskit, pennylane; print('Environment OK')"
```

## 6. Prepare Datasets

Place datasets under `datasets/` following the required structure documented in `datasets/README.md`.

## 7. Quick Validation Run

```bash
python verify_models.py
```

If this command completes successfully, the environment is ready for full experiments.

## Optional: IBM Quantum Access

To run real-hardware tests:
- Create an IBM Quantum account.
- Configure credentials as described in `test_qiskit_real_hardware.py` help output.
- Start with `--estimate-only` before submitting jobs.

## Troubleshooting

- Import errors: ensure the virtual environment is activated.
- Slow training: reduce batch size and use smaller backbones (for example, `resnet18`).
- Out-of-memory: lower batch size and close other GPU processes.
