# HistoKai

**基于 Tile 的强化学习环境，用于全切片图像（WSI）上的肿瘤检测**


HistoKai 实现了一个兼容 Gymnasium 的 RL 环境，将数字病理学中的肿瘤检测问题转化为导航任务。Agent 在 WSI 的预提取 Tile 网格上移动，仅凭局部组织学特征学习定位肿瘤区域——推理时无需直接访问肿瘤位置标签。

本项目涵盖从原始 WSI 预处理到 RL 训练的完整 Pipeline：

```
Raw WSI (.tif) → Tiling (20×/10×) → ResNet18 Embedding → HDF5 Database → Gymnasium Env → RL Agent (PPO)
```

基于 [Camelyon16](https://camelyon16.grand-challenge.org/) 数据集（乳腺癌淋巴结转移检测）构建，这是 WSI 分析领域的标准 Benchmark。

---

## 环境概述：`wsi_env.py`

`WSIEnv` 是一个自定义 [Gymnasium](https://gymnasium.farama.org/) 环境，Agent 在由单张 WSI 构建的 Tile 网格世界中进行导航。

**核心设计：**

| 组件 | 说明 |
|------|------|
| **Grid World** | 20× 放大倍率下的 Tile（224×224 px）构成二维网格；仅组织区域的 Tile 可通行 |
| **Observation** | 1660 维向量 = 512×3（20×、10× 及 Thumbnail Embedding）+ 2（归一化坐标）+ 121（11×11 局部 Visited Map）+ 1（Time Budget） |
| **Action Space** | `Discrete(4)` — 上 / 下 / 左 / 右（可选 `Discrete(5)` 含 STOP 动作） |
| **Reward** | 可配置：步骤惩罚、背景/重访惩罚，到达肿瘤区域时获得较大正 Reward |
| **Starting Modes** | `fixed` · `distance_band` · `random_tissue` — 支持 Curriculum Learning |
| **Embeddings** | 通过自监督和 ImageNet 预训练 ResNet18 预先计算 |

环境从每张 WSI 对应的单个 HDF5 文件加载所有数据，实现快速 Reset，Episode 过程中无磁盘 I/O 开销。

> **教程 →** 参见 [WSI_Env_Tutorial.ipynb](WSI_Env_Tutorial.ipynb)，提供逐步介绍，涵盖环境创建、Observation 分解、网格可视化、Reward 机制、起始策略及 Stable-Baselines3 集成。

---

## Notebooks（实现、实验与分析）

### (1) Tile 可视化

[`(1)tile_visualization.ipynb`](<(1)tile_visualization.ipynb>)

可视化 WSI 的分块过程。使用 OpenSlide 打开 `.tif` 切片，在 20× 放大倍率下叠加 Tile 网格，并检查单个 Tile。帮助直观理解 WSI 如何被分解为构成 RL 环境的网格。

### (2) WSI 预处理

[`(2)wsi_preprocessing.ipynb`](<(2)wsi_preprocessing.ipynb>)

核心数据 Pipeline Notebook。对 Camelyon16 训练集中的每张 WSI 执行以下操作：

1. **Tiling** — 在 20× 下提取 224×224 Tile，并提取 10× 下对应的 Context Patch
2. **Tissue Masking** — 生成二值组织 Mask，并调整至 Tile 网格分辨率
3. **Tumor Masking** — 解析标注 XML 文件，创建每个 Tile 的肿瘤标签（面积阈值 0.3）
4. **Embedding** — 使用自监督和 ImageNet 预训练 ResNet18 分别计算 512 维特征向量
5. **HDF5 Storage** — 将所有 Embedding、坐标、Mask 及 Metadata 写入每张切片对应的 `.h5` 文件

生成供 `WSIEnv` 使用的 `tile_database/tumor_*.h5` 文件。

### (3) 标注与 Tile 可视化

[`(3)annotation_tile_visualization.ipynb`](<(3)annotation_tile_visualization.ipynb>)

在 Tile 网格上叠加肿瘤标注。对于给定的 WSI：

- 展示带有肿瘤区域轮廓的完整切片
- 放大每个肿瘤区域，在网格上显示 Tile 级别的肿瘤/正常标签
- 提取 Tile Embedding 并生成 UMAP 降维图，可视化肿瘤与正常 Tile 在特征空间中的分离情况

用于验证标注质量，并评估 Embedding 空间是否具有判别信号。

### (4) HDF5 数据库测试

[`(4)h5_database_test.ipynb`](<(4)h5_database_test.ipynb>)

对 Notebook (2) 生成的 HDF5 数据库进行完整性检查。检查 Dataset 形状、属性值、Embedding 统计信息和 Mask 分布，在进行 RL 训练之前确认数据完整性。

### (5) RL 设置与单张 WSI

[`(5)RL_setup_and_Single_WSI.ipynb`](<(5)RL_setup_and_Single_WSI.ipynb>)

第一个 RL 实验。在**单张 WSI 上以固定起始位置**（距肿瘤边界 3–5 步 BFS 距离）训练 PPO Agent。验证以下内容：

- Gymnasium 环境正确连接（通过 `check_env`）
- RL Pipeline 收敛——Reward 增加，Episode 长度减少
- Agent 学会短程导航至肿瘤区域

这作为 Baseline 验证，证明在扩展到更难设置之前，环境和训练循环能正常工作。

### (6) Single WSI Curriculum

[暂未提供]

从固定起点扩展到同一张 WSI 上的随机起始位置，测试 Agent 能否从越来越远的距离导航至肿瘤。比较两种训练策略：

- **Sequential Curriculum**（第 1–7 节）：分阶段训练——2a（3–5 BFS 步）→ 2b（10–20）→ 2c（全部组织 Tile）——在成功率达到 70% 时自动晋级。结果：*灾难性遗忘*——后续阶段会破坏先前学到的 Policy，最终模型在三个距离 Pool 上的成功率分别降至 32%/8%/2%。
- **Mixed-Distance Training**（第 8 节）：在单一训练阶段从所有组织 Tile 均匀采样起始位置，使用更宽的网络（256-256）。数值略好（48%/20%/6%），但 Trajectory 分析显示 Agent 仅学会了固定的"向下走"策略——所有成功 Episode 都是直线垂直路径。

核心发现：Observation Space（Tile Embedding + 坐标 + Visited Map）编码的是"组织长什么样"，而非"肿瘤相对于 Agent 的方向"。在 Camelyon16 上，正常组织的 Embedding 无论距肿瘤多近都几乎无法区分，导致 Agent 没有方向信号可以学习。这是一个信息论层面的瓶颈，而非训练策略问题。

### (6_1) Single WSI Curriculum — 对照实验

[暂未提供]

作为 Notebook (6) 的**控制变量后续实验**，旨在排除失败的其他解释：

| (6) 中的潜在干扰因素 | (6_1) 的应对方式 |
|-------------------|----------------|
| 距离跨度过大（3-5 → 10-20，跳过 5-10） | 平滑渐进：3-5 → 5-10 → 10-20 |
| 每阶段训练步数过少（20 万–50 万） | 每阶段 100 万步（共 300 万步） |
| (6) 第 8 节同时改变了多个变量 | 所有超参数/网络结构不变，仅调整距离粒度和训练预算 |

此外引入了系统性的跨阶段遗忘检测（每个模型在所有前序距离 Pool 上评估）和批量 Trajectory 导出（每阶段 30 张 PNG 用于逐 Episode 检查）。

结果：Agent 在每个阶段仍只学会了固定方向 Policy（如 2a 阶段"始终向上"，2b 阶段"始终向左"）。跨阶段成功率集中在 35–58%，无实质性改善。这明确排除了"训练不足"的替代假说，进一步证实了 (6) 中识别的 Observation Space 信息瓶颈。

### (7) RecurrentPPO

[暂未提供]

测试**序列记忆**（LSTM）能否从观测历史中提取方向信号。使用 RecurrentPPO（[SB3-Contrib](https://sb3-contrib.readthedocs.io/)），采用与 (6) 第 8 节相同的 Mixed-Distance 训练设置。

| 模型 | 网络结构 | 参数量 | 2a (3-5) | 2b (10-20) | 2c (全组织) |
|------|---------|--------|----------|------------|------------|
| MLP-PPO（NB6） | [256,256] MLP | ~1M | 48% | 20% | 6% |
| RecurrentPPO | [256,256] + LSTM(256) | 4.2M | 54% | 28% | 20% |

RecurrentPPO 在所有距离范围上均提升了成功率（2c 相对提升最显著，达 +233%）。然而 Trajectory 可视化显示相同模式：成功 Episode 仍然是向下的直线路径。LSTM 使 Agent 能更稳定地保持选定方向（减少轨迹中途的震荡），但并未实现真正的方向性导航。

结论：失败并非源于缺乏时序记忆——即使有完整的观测历史，LSTM 在该数据集上也无法从 Embedding 空间中提取方向梯度。瓶颈在于 Observation 的内容，而非 Policy 的网络结构。

---

## HDF5 数据库结构

每个 `tile_database/{slide_id}.h5` 遵循以下结构：

```
{slide_id}.h5
├── embeddings_20x_s   (N, 512) float32   # 自监督 ResNet18，20× Tile
├── embeddings_10x_s   (N, 512) float32   # 自监督 ResNet18，10× Context
├── embeddings_20x_i   (N, 512) float32   # ImageNet 预训练 ResNet18，20× Tile
├── embeddings_10x_i   (N, 512) float32   # ImageNet 预训练 ResNet18，10× Context
├── coords             (N, 2)   int32     # 每个 Tile 的 Level-0 像素坐标
├── tissue_mask        (N,)     bool      # 是否为组织？
├── tumor_mask         (N,)     bool      # 是否为肿瘤？（面积阈值 0.3）
├── thumbnail          (H,W,3)  uint8     # 低分辨率切片 Thumbnail
├── thumbnail_embedding_s  (512,) float32 # Thumbnail Embedding（自监督）
├── thumbnail_embedding_i  (512,) float32 # Thumbnail Embedding（ImageNet）
└── attrs
    ├── tile_size          224
    ├── level_20x          int
    ├── level_10x          int
    ├── mpp                float
    └── slide_dimensions   (W, H)
```

---

## 环境配置

```bash
conda env create -f environment.yml
conda activate wsi-rl
```

**数据：** 将 Camelyon16 训练切片（`.tif`）放置在 `data/camelyon16/training/tumor/` 目录下，标注文件（`.xml`）放置在 `data/camelyon16/annotations/` 目录下。运行 Notebook (2) 生成 HDF5 Tile 数据库（运行 Notebook (3) 以添加 `tumor_mask: (N,) bool` 字段）。

**预训练权重（ResNet18）：** 通常使用 ImageNet 预训练权重。如需领域专属预训练，请将自监督 ResNet18 Checkpoint 下载至 `pre_trained_resnet/self-supervised-histopathology/pytorchnative_tenpercent_resnet18.ckpt`（来源：[ozanciga/self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology)）。

---
