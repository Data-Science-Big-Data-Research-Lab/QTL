# Contributing

Thank you for contributing.

## Development Principles

- Keep changes focused and traceable.
- Prefer reproducible experiments.
- Avoid modifying multiple concerns in one pull request.
- Preserve compatibility with existing training scripts unless a change is justified.

## Branching and Pull Requests

1. Create a feature branch from `main`.
2. Implement and test your changes.
3. Open a pull request with:
   - clear problem statement
   - implementation summary
   - validation evidence

## Code Style

- Follow existing project style.
- Use descriptive variable names.
- Keep comments concise and technical.
- Add docstrings for non-trivial functions.

## Testing Expectations

Before submitting, run:

```bash
python verify_models.py
```

If you changed experiment orchestration, also run:

```bash
python run_paper_experiments.py --dry-run
```

## Documentation

Update relevant `.md` files when behavior, arguments, or outputs change.

## Reporting Issues

Please include:
- operating system
- Python version
- command used
- full error trace
- steps to reproduce
