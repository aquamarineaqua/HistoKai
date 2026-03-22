# HistoKai Notebook Overview

This repository contains a WSI preprocessing and visualization workflow for the Camelyon16 dataset.

## Notebook Roles

### 1. (1)tile_visualization.ipynb
- Validates the Python environment and key dependencies (OpenSlide, PyTorch, Gymnasium, SB3).
- Demonstrates tile extraction across magnifications and coordinate mapping logic.
- Builds tissue masks (HSV thresholding + morphology + connected-components filtering).
- Tests embedding extraction and basic embedding visualization (t-SNE/UMAP).

### 2. (2)wsi_preprocessing.ipynb
- Main preprocessing pipeline for Camelyon16 tumor WSIs.
- Extracts 20x and 10x tile embeddings and computes tile-level tissue masks.
- Saves one H5 file per slide with embeddings, coordinates, masks, thumbnails, and metadata.
- Includes both single-slide testing and batch processing for all tumor slides.

### 3. (3)annotation_tile_visualization.ipynb
- Parses Camelyon16 XML annotations and maps tumor polygons to tile-level labels.
- Computes tumor overlap ratio per tile and assigns tumor/normal labels.
- Visualizes tumor regions on thumbnails and tile grids with adaptive region cropping.
- Performs embedding-space visualization (UMAP/t-SNE) for tumor-vs-normal comparison.

### 4. (4)h5_database_test.ipynb
- Validates the generated H5 database structure and metadata.
- Inspects tissue and tumor masks and summarizes tile statistics.
- Provides full-slide and cropped tumor-region visual checks for quick QA.

## Environment Setup

This project provides a Conda environment file: `environment.yml`.

### 1. Create environment

```bash
conda env create -f environment.yml
```

### 2. Activate environment

```bash
conda activate wsi-rl
```

### 3. Open notebooks with the correct kernel
- In VS Code/Jupyter, select the `wsi-rl` kernel before running notebook cells.

## Key Dependencies

- Python 3.10
- OpenSlide (`openslide` + `openslide-python`)
- PyTorch + torchvision (CUDA 12.1 in the current environment file)
- h5py, numpy, scipy, scikit-learn, pandas
- OpenCV, Pillow
- matplotlib, seaborn, jupyterlab, ipywidgets
- stable-baselines3, sb3-contrib, gymnasium
