# Git Command Reference

This file provides a compact set of Git commands used in this project workflow.

## Repository Setup

```bash
git clone https://github.com/Data-Science-Big-Data-Research-Lab/QTL.git
cd QTL
git checkout main
git pull origin main
```

## Create a Working Branch

```bash
git checkout -b feature/<short-description>
```

## Inspect Changes

```bash
git status
git diff
git log --oneline --graph --decorate -n 20
```

## Stage and Commit

```bash
git add <file_or_folder>
git commit -m "<clear commit message>"
```

## Keep Branch Updated

```bash
git fetch origin
git rebase origin/main
```

## Push Branch

```bash
git push -u origin feature/<short-description>
```

## Open Pull Request

Use your Git hosting platform to open a PR from your feature branch to `main`.

## Undo Local Changes (Safe Cases)

Unstage files:

```bash
git restore --staged <file>
```

Discard local modifications to a file:

```bash
git restore <file>
```

## Tagging a Release

```bash
git checkout main
git pull origin main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## Notes

- Use atomic commits.
- Write imperative commit messages.
- Rebase before opening a PR when possible.
