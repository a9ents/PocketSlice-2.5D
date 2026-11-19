# PocketSlice — 厚度编码的 2.5D 配体结合位点检测

> Thickness-encoded 2.5D representation for ligand-binding-site detection,
> formulated as center regression with a tiny U-Net + CenterNet.

PocketSlice 把医学影像中的 **2.5D 切片范式**迁移到蛋白质空腔检测：沿蛋白主轴
切一组 2D 平面，每个平面由三个通道编码 —— 到蛋白原子的距离、沿法线的
**空腔厚度（thickness channel）**、原子化学类型。模型是一个轻量 U-Net，
用 CenterNet 范式回归口袋中心热图与尺度，推理时把多轴切片的峰值反投影回
3D、投票并用 NMS 聚合，输出按分数排序的候选中心列表。

核心思想：**厚度通道保留了 3D 的方向性深度信息，却只需 2.5D 的计算成本**；
把"检测"改写为"中心回归"，让定位精度显著优于体素分割类方法。

本仓库包含论文方法的核心实现，以及平台/评估所需的全部模型资产文件
（几何权重、重排序权重、功能分类器、数据划分），用于复现 COACH420
留出集上的结果。

---

## 仓库内容

| 文件 | 作用 |
|---|---|
| `model.py` | `TinyUNet`：3 通道输入、2 通道输出（热图 + 尺度）的轻量 U-Net |
| `slicing.py` | 2.5D 切片生成与**厚度通道**计算，CenterNet 标签生成 |
| `pdb_utils.py` | PDB 解析、原子化学特征、PCA 主轴、配体质心 |
| `eval_optimized.py` | 推理与评估：峰值提取、多轴投票 + NMS、命中率统计 |
| `train_large.py` | 训练脚本（CenterNet focal loss + L1 scale loss） |
| `tiny_unet_centernet.pt` | 第一层几何模型权重（对应 `TinyUNet(in_ch=3, out_ch=2, base=32)`）；GUI 中"PocketSlice 权重(.pt)" |
| `dataset_split.pt` | COACH420 的 336/84 训练/测试划分（PDB ID → 配体 resname）；非神经网络权重，请勿删除 |
| `pocket_function_rf_model.pkl` | 第二层功能分类器（RandomForest，150 树 / 21 特征）；GUI 中"功能分类器(.pkl)"，用 `joblib.load` 读取 |
| `reranker_model.pth` | 第三层重排序模型（torch state_dict，encoder + cross-attention）；GUI 中"重排序模型(.pth)" |

> 注：早期实验脚本（2 通道单头的 `train.py`、3D/对比学习等）与最终模型无关，未纳入本仓库。
>
> 模型资产加载方式：
> ```python
> # 几何权重（state_dict）
> model.load_state_dict(torch.load("tiny_unet_centernet.pt", map_location="cpu", weights_only=True))
> # 功能分类器（joblib 序列化的 RandomForest，不能用 pickle.load）
> import joblib
> clf = joblib.load("pocket_function_rf_model.pkl")
> # 重排序器（state_dict）
> sd = torch.load("reranker_model.pth", map_location="cpu", weights_only=False)
> # 数据划分
> split = torch.load("dataset_split.pt", map_location="cpu", weights_only=False)
> ```

---

## 环境安装

```bash
conda create -n pocketslice python=3.10
conda activate pocketslice
pip install -r requirements.txt
```

依赖仅为数值计算、结构解析与深度学习基础库；无 Vina / P2Rank / Fpocket 等
外部工具依赖。

---

## 数据准备

推理与训练都需要 COACH420 的 PDB 结构。请自行获取 COACH420 数据集，把
420 个复合物 PDB 放入 `coach420_pdbs/`（文件名格式 `1abcA.pdb`，即 PDB ID
后接链 ID）。`dataset_split.pt` 指定的配体 resname 必须与 PDB 中的 HETATM
残基名一致。

```
PocketSlice-2.5D/
├── coach420_pdbs/        # ← 自行下载放置（不随仓库分发）
│   ├── 148lE.pdb
│   └── ...
└── ...
```

`dataset_split.pt` 的结构：

```python
{"train": {"2d29": "FAD", ...},   # 336 个
 "test":  {"2zu3": "ZU3", ...}}   # 84 个
```

---

## 快速复现（推理评估）

在 84 个留出蛋白上评估，阈值 0.2：

```bash
python eval_optimized.py --threshold 0.2
```

常用参数：

```bash
python eval_optimized.py --limit 20            # 只跑前 20 个，快速冒烟测试
python eval_optimized.py --zero-channels 1     # 消融：置零厚度通道（0=dist,1=thick,2=chem）
python eval_optimized.py --no-dump             # 不导出预测 JSON
```

脚本会打印 Top-1 / Top-3 / Top-10 命中率与 Oracle@10，并把候选中心导出到
`preds_ours.json`。

---

## 训练（可选）

`train_large.py` 从预生成的切片缓存训练。先用 `slicing.generate_slices`
把训练集蛋白切片缓存到 `cache_slices/<pdb_id>.pt`（每个文件含 `inputs` /
`heatmaps` / `scales`），然后：

```bash
python train_large.py
```

最佳权重按验证 loss 保存到 `tiny_unet_centernet.pt`。仓库内已附带在完整
训练集上训练好的权重，通常无需重训。

---

## 方法要点

- **三通道 2.5D 表示**：距离通道（到最近蛋白原子）、**厚度通道**（沿切片法线
  方向的空腔 spanned 厚度，核心创新）、化学通道（C/S 疏水 = +1，O/N 亲水 = −1）。
- **CenterNet 目标**：高斯热图回归中心，L1 回归口袋尺度；无口袋的切片热图
  标签为 −1，不计入损失。
- **3D 聚合**：多主轴切片预测经反投影得到 3D 候选，跨轴距离 < 3 Å 视为投票，
  < 4 Å 强制 NMS 合并，最终按（票数, 分数）降序排序。

---

## 引用

如使用本代码，请引用 PocketSlice 原文（投稿中，待补充 DOI）。

## 许可

MIT License —— 详见 `MIT License` 文件。
