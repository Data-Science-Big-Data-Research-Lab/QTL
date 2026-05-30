# Dataset Download Instructions

This folder should contain the datasets used in the experiments. Due to size constraints, the raw dataset files are excluded from this repository. You can download them using the links and instructions below:

## 1. Hymenoptera (Ants vs Bees)
- **Description**: Binary classification of ants and bees (~240 images).
- **Download Link**: [hymenoptera_data.zip](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
- **Setup**: You can download it automatically by running:
  ```bash
  python data/download_datasets.py --dataset hymenoptera
  ```
  Or extract it manually to `data/datasets/hymenoptera/` such that you have `data/datasets/hymenoptera/train/` and `data/datasets/hymenoptera/test/`.

## 2. Brain Tumor MRI
- **Description**: Brain tumor classification using MRI scans (glioma, meningioma, pituitary, no tumor).
- **Download Link**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Setup**:
  1. Download the dataset from Kaggle.
  2. Extract it to `data/datasets/brain_tumor/`.
  3. Ensure the structure is:
     - `data/datasets/brain_tumor/Training/` (containing class subfolders)
     - `data/datasets/brain_tumor/Testing/` (containing class subfolders)

## 3. Cats vs Dogs
- **Description**: Cat vs Dog binary classification dataset.
- **Download Link**: [Microsoft Cats vs Dogs Dataset](https://www.microsoft.com/download/confirmation.aspx?id=54765)
- **Setup**:
  1. Download and extract the dataset.
  2. Organize the images into:
     - `data/datasets/cats_vs_dogs/train/cats/` and `data/datasets/cats_vs_dogs/train/dogs/`
     - `data/datasets/cats_vs_dogs/val/cats/` and `data/datasets/cats_vs_dogs/val/dogs/`

## 4. Solar Panel Dust Detection
- **Description**: Detection of dust/soiling on solar panels.
- **Setup**:
  1. Place the clean and dusty images under:
     - `data/datasets/solar_dust/train/clean/` and `data/datasets/solar_dust/train/dusty/`
     - `data/datasets/solar_dust/val/clean/` and `data/datasets/solar_dust/val/dusty/`

---
Alternatively, you can run the helper script:
```bash
python data/download_datasets.py --list
```
to display detailed instructions and status check for all datasets.
