# HistoKai

**A Tile-Based Reinforcement Learning Environment for Tumor Detection on Whole Slide Images**

<img src="Cover/Cover.png" alt="HistoKai Logo" style="zoom:50%;" />

HistoKai implements a Gymnasium-compatible RL environment that formulates tumor detection in digital pathology as a navigation task. An agent moves across a grid of pre-extracted tiles on a Whole Slide Image (WSI), learning to locate tumor regions using only local histological features — without direct access to tumor location labels at inference time.

The project covers the full pipeline from raw WSI preprocessing to RL training:

```
Raw WSI (.tif) → Tiling (20×/10×) → ResNet18 Embedding → HDF5 Database → Gymnasium Env → RL Agent (PPO)
```

Built on the [Camelyon16](https://camelyon16.grand-challenge.org/) dataset (breast cancer lymph node metastasis detection), a standard WSI benchmark.

---

## Environment Overview: `wsi_env.py`

`WSIEnv` is a custom [Gymnasium](https://gymnasium.farama.org/) environment where the agent navigates a tile-grid world derived from a single WSI.

**Key design:**

| Component | Description |
|-----------|-------------|
| **Grid World** | 20× magnification tiles (224×224 px) on a 2D grid; only tissue tiles are traversable |
| **Observation** | 1660-d vector = 512×3 (20× + 10× + thumbnail embeddings) + 2 (normalized coords) + 121 (11×11 local visited map) + 1 (time budget) |
| **Action Space** | `Discrete(4)` — up / down / left / right (optionally `Discrete(5)` with a STOP action) |
| **Reward** | Configurable: step penalty, background/revisit penalties, large positive reward upon reaching tumor |
| **Starting Modes** | `fixed` · `distance_band` · `random_tissue` — supports curriculum learning |
| **Embeddings** | Pre-computed via Self-Supervised and ImageNet-pretrained ResNet18 |

The environment loads all data from a single HDF5 file per WSI, enabling fast reset with zero disk I/O during episodes.

> **Tutorial →** See [WSI_Env_Tutorial.ipynb](WSI_Env_Tutorial.ipynb) for a step-by-step walkthrough covering environment creation, observation decomposition, grid visualization, reward mechanics, starting strategies, and Stable-Baselines3 integration.

---

## Notebooks (Implementation & Experiments & Analysis)

### (1) Tile Visualization

[`(1)tile_visualization.ipynb`](<(1)tile_visualization.ipynb>)

Visualizes the WSI tiling process — opens a `.tif` slide with OpenSlide, overlays the tile grid at 20× magnification, and inspects individual tiles.

### (2) WSI Preprocessing

[`(2)wsi_preprocessing.ipynb`](<(2)wsi_preprocessing.ipynb>)

Core data pipeline. Extracts 224×224 tiles at 20×/10×, generates tissue & tumor masks, computes ResNet18 embeddings (Self-Supervised + ImageNet), and writes everything into one `.h5` file per slide. Produces the `tile_database/tumor_*.h5` files consumed by `WSIEnv`.

### (3) Annotation & Tile Visualization

[`(3)annotation_tile_visualization.ipynb`](<(3)annotation_tile_visualization.ipynb>)

Overlays tumor annotations on the tile grid, zooms into tumor regions with tile-level labels, and produces UMAP plots of tile embeddings. Useful for verifying annotation quality and embedding discriminability.

### (4) HDF5 Database Test

[`(4)h5_database_test.ipynb`](<(4)h5_database_test.ipynb>)

Sanity-checks the HDF5 databases produced by notebook (2) — inspects dataset shapes, attribute values, embedding statistics, and mask distributions.

### (5) RL Setup & Single WSI

[`(5)RL_setup_and_Single_WSI.ipynb`](<(5)RL_setup_and_Single_WSI.ipynb>)

First RL experiment. Trains a PPO agent on a single WSI with fixed starting positions near the tumor boundary (3–5 BFS steps). Serves as the baseline proof that the environment and training loop work correctly.

### (6) Single WSI Curriculum

[`(6)Single_WSI_Curriculum.ipynb`](<(6)Single_WSI_Curriculum.ipynb>)

Compares **sequential curriculum** (staged distance increase, suffers catastrophic forgetting) and **mixed-distance training** (uniform sampling from all tissue) on a single WSI. Core finding: the observation space lacks directional signal toward tumor — normal tile embeddings are nearly indistinguishable regardless of proximity, creating an information-theoretic bottleneck.

### (6_1) Single WSI Curriculum — Control Experiment

[`(6_1)Single_WSI_Curriculum1.ipynb`](<(6_1)Single_WSI_Curriculum1.ipynb>)

Controlled follow-up to notebook (6): uses finer distance progression (3-5 → 5-10 → 10-20) and 3× more training budget to rule out inadequate training as an explanation. Results confirm the observation space bottleneck — the agent still learns only fixed-direction policies.

### (7) RecurrentPPO

[`(7)RecurrentPPO_Curriculum.ipynb`](<(7)RecurrentPPO_Curriculum.ipynb>)

Tests whether LSTM memory (RecurrentPPO) can extract directional signal from observation history. Improves success rates over MLP-PPO (e.g., 6% → 20% on all-tissue starts), but trajectory analysis shows the same straight-line strategy. Confirms the bottleneck lies in observation content, not policy architecture.

---

## HDF5 Database Schema

Each `tile_database/{slide_id}.h5` follows this structure:

```
{slide_id}.h5
├── embeddings_20x_s   (N, 512) float32   # Self-Supervised ResNet18, 20× tiles
├── embeddings_10x_s   (N, 512) float32   # Self-Supervised ResNet18, 10× context
├── embeddings_20x_i   (N, 512) float32   # ImageNet-pretrained ResNet18, 20× tiles
├── embeddings_10x_i   (N, 512) float32   # ImageNet-pretrained ResNet18, 10× context
├── coords             (N, 2)   int32     # Level-0 pixel coordinates per tile
├── tissue_mask        (N,)     bool      # Is tissue?
├── tumor_mask         (N,)     bool      # Is tumor? (area threshold 0.3)
├── thumbnail          (H,W,3)  uint8     # Low-res slide thumbnail
├── thumbnail_embedding_s  (512,) float32 # Thumbnail embedding (Self-Supervised)
├── thumbnail_embedding_i  (512,) float32 # Thumbnail embedding (ImageNet)
└── attrs
    ├── tile_size          224
    ├── level_20x          int
    ├── level_10x          int
    ├── mpp                float
    └── slide_dimensions   (W, H)
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate wsi-rl
```

**Data:** Place Camelyon16 training slides (`.tif`) in `data/camelyon16/training/tumor/` and annotations (`.xml`) in `data/camelyon16/annotations/`. Run notebook (2) to generate the HDF5 tile databases (and notebook (3) to add `tumor_mask: (N,) bool` field).

**Pre-trained weights (ResNet18):** Normally we use ImageNet-pretrained weights. With domain-specific pretraining, download the self-supervised ResNet18 checkpoint to `pre_trained_resnet/self-supervised-histopathology/pytorchnative_tenpercent_resnet18.ckpt` (from [ozanciga/self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology)).

---
