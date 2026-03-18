# Paper Experiment Runner

This file documents the scripts used to generate reproducible results for the paper.

## Main Scripts

- `verify_models.py`: quick pre-flight validation.
- `run_paper_experiments.py`: systematic experiments with logging and CSV output.
- `run_complete_benchmark.py`: extended benchmark matrix across multiple settings.

## Recommended Workflow

1. Validate setup:

```bash
python verify_models.py
```

2. Execute paper experiments:

```bash
python run_paper_experiments.py
```

3. Optional large benchmark:

```bash
python run_complete_benchmark.py
```

## Dry Run Mode

Inspect planned experiments without execution:

```bash
python run_paper_experiments.py --dry-run
```

## Configuration Overview

Default paper configuration is defined inside `run_paper_experiments.py` and includes:
- 4 datasets
- 4 CNN backbones
- multiple training approaches (classical and hybrid)
- fixed training budget per experiment

## Outputs

- CSV summary with accuracy and timing metrics
- execution logs with progress and failure details
- per-run model checkpoints and plot artifacts

## Operational Notes

- Start with `verify_models.py` to catch environment issues early.
- Prefer `resnet18` for quick troubleshooting.
- For long runs, monitor disk space and GPU memory.
- Use `--output-dir` to separate result sets by date or experiment group.
