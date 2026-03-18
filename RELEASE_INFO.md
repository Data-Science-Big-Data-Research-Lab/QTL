# Release Information

## Current Status

This repository contains the research code used for the Quantum Transfer Learning paper experiments.

## Included Components

- Classical transfer learning baseline training
- Hybrid Qiskit training (ideal and noisy)
- Hybrid PennyLane training (ideal and noisy)
- Paper experiment orchestration and benchmarking
- Optional IBM Quantum real-hardware evaluation

## Output Artifacts

- model checkpoints (`model_saved/`)
- metrics and plots (`static/metrics/`)
- experiment logs and CSV summaries

## Reproducibility

To reproduce core results:

1. Install dependencies (`INSTALLATION.md`).
2. Prepare datasets (`datasets/README.md`).
3. Run quick validation (`verify_models.py`).
4. Execute paper experiments (`run_paper_experiments.py`).

## Compatibility Notes

- Python 3.8+
- Requires versions from `requirements.txt`
- Real-hardware execution depends on valid IBM Quantum access

## Known Operational Constraints

- Noisy hybrid runs are slower than classical baselines.
- Real-hardware jobs may queue depending on backend availability.
- Large experiment grids require substantial compute time and storage.

## Versioning Policy

Use semantic tags in the form `vMAJOR.MINOR.PATCH`.
- MAJOR: incompatible changes
- MINOR: backward-compatible features
- PATCH: bug fixes and documentation updates
