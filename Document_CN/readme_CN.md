# HistoKai

**基于 Tile 的强化学习环境，用于全切片图像上的肿瘤检测**

HistoKai 实现了一个兼容 Gymnasium 的强化学习环境，将数字病理学中的肿瘤检测问题转化为导航任务。智能体在全切片图像（WSI）的预提取Tile网格上移动，仅凭局部组织学特征学习定位肿瘤区域——在推理时无需直接访问肿瘤位置标签。

本项目涵盖从原始 WSI 预处理到强化学习训练的完整流程：

```
原始 WSI (.tif) → 分块 (20×/10×) → ResNet18 嵌入 → HDF5 数据库 → Gymnasium 环境 → RL 智能体 (PPO)
```

基于 [Camelyon16](https://camelyon16.grand-challenge.org/) 数据集（乳腺癌淋巴结转移检测）构建。

---

## 环境概述：`wsi_env.py`

`WSIEnv` 是一个自定义 [Gymnasium](https://gymnasium.farama.org/) 环境，智能体在由单张 WSI 构建的Tile网格世界中进行导航。

**核心设计：**

| 组件 | 说明 |
|------|------|
| **网格世界** | 20× 放大倍率下的Tile（224×224 像素）构成二维网格；仅组织区域的Tile可通行 |
| **观测空间** | 1660 维向量 = 512×3（20×、10× 及缩略图嵌入）+ 2（归一化坐标）+ 121（11×11 局部已访问地图）+ 1（时间预算） |
| **动作空间** | `Discrete(4)` — 上 / 下 / 左 / 右（可选 `Discrete(5)` 含停止动作） |
| **奖励** | 可配置：步骤惩罚、背景/重访惩罚，到达肿瘤区域时获得较大正奖励 |
| **起始模式** | `fixed`（固定）· `distance_band`（距离带）· `random_tissue`（随机组织）— 支持课程学习 |
| **嵌入** | 通过自监督和 ImageNet 预训练 ResNet18 预先计算 |

环境从每张 WSI 对应的单个 HDF5 文件加载所有数据，实现快速重置，训练过程中无磁盘 I/O 开销。

> **教程 →** 参见 [WSI_Env_Tutorial.ipynb](WSI_Env_Tutorial.ipynb)，提供逐步介绍，涵盖环境创建、观测分解、网格可视化、奖励机制、起始策略及 Stable-Baselines3 集成。

---

## Notebooks

### (1) Tile可视化 — [`(1)tile_visualization.ipynb`](<(1)tile_visualization.ipynb>)

可视化 WSI 的分块过程。使用 OpenSlide 打开 `.tif` 切片，在 20× 放大倍率下叠加Tile网格，并检查单个Tile。帮助直观理解 WSI 如何被分解为构成 RL 环境的网格。

### (2) WSI 预处理 — [`(2)wsi_preprocessing.ipynb`](<(2)wsi_preprocessing.ipynb>)

核心数据流水线 Notebook。对 Camelyon16 训练集中的每张 WSI 执行以下操作：

1. **分块** — 在 20× 下提取 224×224 Tile，并提取 10× 下对应的上下文图像块
2. **组织掩码** — 生成二值组织掩码，并调整至Tile网格分辨率
3. **肿瘤掩码** — 解析标注 XML 文件，创建每个Tile的肿瘤标签（面积阈值 0.3）
4. **嵌入** — 使用自监督和 ImageNet 预训练 ResNet18 分别计算 512 维特征向量
5. **HDF5 存储** — 将所有嵌入、坐标、掩码及元数据写入每张切片对应的 `.h5` 文件

生成供 `WSIEnv` 使用的 `tile_database/tumor_*.h5` 文件。

### (3) 标注与Tile可视化 — [`(3)annotation_tile_visualization.ipynb`](<(3)annotation_tile_visualization.ipynb>)

在Tile网格上叠加肿瘤标注。对于给定的 WSI：

- 展示带有肿瘤区域轮廓的完整切片
- 放大每个肿瘤区域，在网格上显示Tile级别的肿瘤/正常标签
- 提取Tile嵌入并生成 UMAP 降维图，可视化肿瘤与正常Tile在特征空间中的分离情况

用于验证标注质量，并评估嵌入空间是否具有判别信号。

### (4) HDF5 数据库测试 — [`(4)h5_database_test.ipynb`](<(4)h5_database_test.ipynb>)

对 Notebook (2) 生成的 HDF5 数据库进行健全性检查。检查数据集形状、属性值、嵌入统计信息和掩码分布，在进行 RL 训练之前确认数据完整性。

### (5) RL 设置与单张 WSI — [`(5)RL_setup_and_Single_WSI.ipynb`](<(5)RL_setup_and_Single_WSI.ipynb>)

第一个 RL 实验。在**单张 WSI 上以固定起始位置**（距肿瘤边界 3–5 步 BFS 距离）训练 PPO 智能体。验证以下内容：

- Gymnasium 环境正确连接（通过 `check_env`）
- RL 流程收敛——奖励增加，回合长度减少
- 智能体学会短程导航至肿瘤区域

这作为基线验证，证明在扩展到更难设置之前，环境和训练循环能正常工作。

---

## HDF5 数据库结构

每个 `tile_database/{slide_id}.h5` 遵循以下结构：

```
{slide_id}.h5
├── embeddings_20x_s   (N, 512) float32   # 自监督 ResNet18，20× Tile
├── embeddings_10x_s   (N, 512) float32   # 自监督 ResNet18，10× 上下文
├── embeddings_20x_i   (N, 512) float32   # ImageNet 预训练 ResNet18，20× Tile
├── embeddings_10x_i   (N, 512) float32   # ImageNet 预训练 ResNet18，10× 上下文
├── coords             (N, 2)   int32     # 每个Tile的 Level-0 像素坐标
├── tissue_mask        (N,)     bool      # 是否为组织？
├── tumor_mask         (N,)     bool      # 是否为肿瘤？（面积阈值 0.3）
├── thumbnail          (H,W,3)  uint8     # 低分辨率切片缩略图
├── thumbnail_embedding_s  (512,) float32 # 缩略图嵌入（自监督）
├── thumbnail_embedding_i  (512,) float32 # 缩略图嵌入（ImageNet）
└── attrs
    ├── tile_size          224
    ├── level_20x          int
    ├── level_10x          int
    ├── mpp                float
    └── slide_dimensions   (W, H)
```

---

## 安装

```bash
conda env create -f environment.yml
conda activate wsi-rl
```

**数据：** 将 Camelyon16 训练切片（`.tif`）放置在 `data/camelyon16/training/tumor/` 目录下，标注文件（`.xml`）放置在 `data/camelyon16/annotations/` 目录下。运行 Notebook (2) 生成 HDF5 Tile数据库（运行 Notebook (3) 以添加 `tumor_mask: (N,) bool` 信息）。

**预训练权重（ResNet18）：** 通常使用 ImageNet 预训练权重。如需领域专属预训练，请将自监督 ResNet18 检查点下载至 `pre_trained_resnet/self-supervised-histopathology/pytorchnative_tenpercent_resnet18.ckpt`（来源：[ozanciga/self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology)）。

---
