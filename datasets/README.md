# Dataset Guidelines

This directory stores all image datasets used by the training scripts.

## Required Structure

Each dataset must follow this pattern:

```text
datasets/<dataset_name>/
  train/<class_name>/
    image_001.jpg
    image_002.jpg
  test/<class_name>/
    image_101.jpg
    image_102.jpg
```

Example:

```text
datasets/hymenoptera/
  train/ants/
  train/bees/
  test/ants/
  test/bees/
```

## Supported Default Dataset Names

- `hymenoptera`
- `brain_tumor`
- `cats_dogs`
- `solar_dust`

You can also add custom datasets if the folder format is respected.

## Image Requirements

- Common formats: JPG, JPEG, PNG, BMP
- RGB images are recommended
- Input images are resized and normalized automatically by the pipeline

## Recommended Size

- Minimum: 100 training images per class, 20 test images per class
- Preferred: 500+ training images per class, 100+ test images per class

## Validation Checklist

- `train/` and `test/` exist
- Each split contains one folder per class
- Class folders are non-empty
- Train and test classes match exactly
