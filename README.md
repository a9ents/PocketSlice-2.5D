# PocketSlice — Thickness-Encoded 2.5D Ligand-Binding Site Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24363235.svg)](https://doi.org/10.5281/zenodo.24363235)

> Thickness-encoded 2.5D representation for ligand-binding-site detection,
> formulated as center regression with a tiny U-Net + CenterNet.

PocketSlice transfers the **2.5D slicing paradigm** from medical imaging to
protein cavity detection: a stack of 2D planes is sliced along the protein's
principal axes, and each plane is encoded with three channels — distance to
protein atoms, **cavity thickness along the slice normal (the thickness
channel)**, and atom chemistry. A lightweight U-Net regresses a center heatmap
and pocket scale in the CenterNet style; at inference, peaks from multiple
axes are back-projected into 3D, voted, and merged by NMS, yielding a
score-ranked list of candidate centers.

The key idea: **the thickness channel preserves directional 3D depth
information at 2.5D computational cost**. By reformulating "detection" as
"center regression," localization accuracy significantly outperforms
voxel-segmentation-style methods.

This repository contains the core implementation of the method together with
all model assets required by the platform and evaluation (geometry weights,
reranker weights, functional classifier, and data split), for reproducing the
results on the COACH420 held-out set.

---

## Repository Contents

| File | Purpose |
|---|---|
| `model.py` | `TinyUNet`: a lightweight U-Net with 3 input channels and 2 output channels (heatmap + scale) |
| `slicing.py` | 2.5D slice generation and **thickness-channel** computation; CenterNet label generation |
| `pdb_utils.py` | PDB parsing, atom-chemical features, PCA principal axes, ligand centroid |
| `eval_optimized.py` | Inference and evaluation: peak extraction, multi-axis voting + NMS, hit-rate statistics |
| `train_large.py` | Training script (CenterNet focal loss + L1 scale loss) |
| `tiny_unet_centernet.pt` | Stage-1 geometry model weights (for `TinyUNet(in_ch=3, out_ch=2, base=32)`); the GUI's "PocketSlice weights (.pt)" |
| `dataset_split.pt` | COACH420 336/84 train/test split (PDB ID → ligand resname); not a neural-network weight — do not delete |
| `pocket_function_rf_model.pkl` | Stage-2 functional classifier (RandomForest, 150 trees / 21 features); the GUI's "functional classifier (.pkl)", read with `joblib.load` |
| `reranker_model.pth` | Stage-3 reranker (torch state_dict, encoder + cross-attention); the GUI's "reranker model (.pth)" |

> Note: early experimental scripts (the 2-channel single-head `train.py`, 3D /
> contrastive-learning variants, etc.) are unrelated to the final model and are
> not included in this repository.
>
> Loading the model assets:
> ```python
> # Geometry weights (state_dict)
> model.load_state_dict(torch.load("tiny_unet_centernet.pt", map_location="cpu", weights_only=True))
> # Functional classifier (joblib-serialized RandomForest; pickle.load will NOT work)
> import joblib
> clf = joblib.load("pocket_function_rf_model.pkl")
> # Reranker (state_dict)
> sd = torch.load("reranker_model.pth", map_location="cpu", weights_only=False)
> # Data split
> split = torch.load("dataset_split.pt", map_location="cpu", weights_only=False)
> ```

---

## Installation

```bash
conda create -n pocketslice python=3.10
conda activate pocketslice
pip install -r requirements.txt
```

Dependencies are limited to numerical/geometry, structure-parsing, and deep
learning basics; no external tools such as Vina / P2Rank / Fpocket are required.

---

## Data Preparation

Both inference and training require the COACH420 PDB structures. Obtain the
COACH420 dataset yourself and place the 420 complex PDB files under
`coach420_pdbs/` (filename format `1abcA.pdb`, i.e. PDB ID followed by chain
ID). The ligand resname specified in `dataset_split.pt` must match the HETATM
residue name in the PDB.

```
PocketSlice-2.5D/
├── coach420_pdbs/        # <- download and place yourself (not distributed)
│   ├── 148lE.pdb
│   └── ...
└── ...
```

Structure of `dataset_split.pt`:

```python
{"train": {"2d29": "FAD", ...},   # 336 entries
 "test":  {"2zu3": "ZU3", ...}}   # 84 entries
```

---

## Quick Reproduction (Inference / Evaluation)

Evaluate on the 84 held-out proteins with a threshold of 0.2:

```bash
python eval_optimized.py --threshold 0.2
```

Common options:

```bash
python eval_optimized.py --limit 20            # run only the first 20, quick smoke test
python eval_optimized.py --zero-channels 1     # ablation: zero out the thickness channel (0=dist, 1=thick, 2=chem)
python eval_optimized.py --no-dump             # do not export prediction JSON
```

The script prints Top-1 / Top-3 / Top-10 hit rates and Oracle@10, and exports
candidate centers to `preds_ours.json`.

---

## Training (Optional)

`train_large.py` trains from pre-generated slice caches. First cache the
training proteins' slices with `slicing.generate_slices` into
`cache_slices/<pdb_id>.pt` (each file contains `inputs` / `heatmaps` /
`scales`), then run:

```bash
python train_large.py
```

The best weights are saved by validation loss to `tiny_unet_centernet.pt`.
Weights trained on the full training set are already included in this
repository, so retraining is usually unnecessary.

---

## Method Highlights

- **Three-channel 2.5D representation**: distance channel (to the nearest
  protein atom), **thickness channel** (the cavity extent spanned along the
  slice normal — the core innovation), and chemistry channel
  (C/S hydrophobic = +1, O/N hydrophilic = −1).
- **CenterNet objective**: Gaussian heatmap regression for the center, L1
  regression for pocket scale; slices without a pocket have a heatmap label
  of −1 and are excluded from the loss.
- **3D aggregation**: predictions from multiple principal axes are
  back-projected to 3D candidates; cross-axis distances < 3 Å count as a
  vote, < 4 Å are forcibly merged by NMS, and candidates are finally ranked
  by (vote count, score) in descending order.

---

## Citation

If you use this code, please cite the software archive (the Zenodo DOI is
registered and citable now):

```bibtex
@software{pocketslice2026,
  author    = {PocketSlice authors},
  title     = {{PocketSlice-2.5D}: Thickness-encoded 2.5D ligand-binding
               site detection},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24363235},
  url       = {https://doi.org/10.5281/zenodo.24363235}
}
```

The accompanying paper is currently under submission; the journal DOI will be
added here upon acceptance.

## License

MIT License — see the `LICENSE` file.
