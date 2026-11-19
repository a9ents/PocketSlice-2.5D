import numpy as np
from scipy.spatial import cKDTree
from pdb_utils import pca_axes

def _compute_thickness(pts_flat, normal, tree, half_thick=10.0, step=0.5, probe=1.7):
    ts = np.arange(-half_thick, half_thick + step, step)
    T = len(ts)
    mid = T // 2
    samples = pts_flat[:, None, :] + ts[None, :, None] * normal[None, None, :]
    d, _ = tree.query(samples.reshape(-1, 3))
    d = d.reshape(pts_flat.shape[0], T)
    occupied = d < probe
    pos_occ = occupied[:, mid:]
    pos_any = pos_occ.any(axis=1)
    pos_first = np.where(pos_any, pos_occ.argmax(axis=1), -1)
    neg_occ = occupied[:, : mid + 1][:, ::-1]
    neg_any = neg_occ.any(axis=1)
    neg_first = np.where(neg_any, neg_occ.argmax(axis=1), -1)
    pos_t = np.where(pos_any, ts[mid + pos_first], 0.0)
    neg_t = np.where(neg_any, ts[mid - neg_first], 0.0)
    return (pos_t - neg_t).astype(np.float32)

def _process_slice_features(pts_flat, n, tree, protein_chem, half_thick, X_shape):
    d, idx = tree.query(pts_flat)
    dist_ch = np.clip(d, 0, 15).reshape(X_shape) / 15.0
    thick = _compute_thickness(pts_flat, n, tree, half_thick=half_thick)
    thick_ch = np.clip(thick, -10, 10).reshape(X_shape) / 10.0
    chem_ch = protein_chem[idx].reshape(X_shape)
    return dist_ch, thick_ch, chem_ch

def generate_slices(protein_coords, protein_chem, ligand_coords, n_per_axis=10, resolution=1.0, half_size=30.0, half_thick=10.0):
    axes = pca_axes(protein_coords)
    center = protein_coords.mean(axis=0)
    proj = (protein_coords - center) @ axes
    tree = cKDTree(protein_coords)
    grid = np.arange(-half_size, half_size + resolution, resolution)
    X, Y = np.meshgrid(grid, grid)

    slices = []
    for axis_idx in range(3):
        lo = np.percentile(proj[:, axis_idx], 5)
        hi = np.percentile(proj[:, axis_idx], 95)
        positions = np.linspace(lo, hi, n_per_axis)
        n = axes[:, axis_idx]
        u = axes[:, (axis_idx + 1) % 3]
        v = axes[:, (axis_idx + 2) % 3]

        for pos in positions:
            plane_center = center + pos * n
            origin = plane_center - half_size * u - half_size * v
            pts = (plane_center[None, None, :] + X[..., None] * u[None, None, :] + Y[..., None] * v[None, None, :])
            
            dist_ch, thick_ch, chem_ch = _process_slice_features(pts.reshape(-1, 3), n, tree, protein_chem, half_thick, X.shape)

            # === CenterNet 范式标签生成 ===
            heatmap = np.full(X.shape, -1.0, dtype=np.float32) # -1 代表该切片无口袋，不计入 Loss
            scale_map = np.zeros(X.shape, dtype=np.float32)

            if len(ligand_coords) > 0:
                rel = ligand_coords - plane_center
                valid = np.abs(rel @ n) < 4.0 # 配体跨越切片的厚度
                if valid.any():
                    lig_2d = np.stack([rel[valid] @ u, rel[valid] @ v], axis=1)
                    lig_center_2d = lig_2d.mean(axis=0)
                    
                    # 1. 生成高斯热图
                    sigma = 2.0
                    dist_2d = np.linalg.norm(np.stack([X, Y], axis=-1) - lig_center_2d, axis=-1)
                    heatmap = np.exp(-(dist_2d**2) / (2 * sigma**2)).astype(np.float32)
                    
                    # 2. 生成尺寸 Scale
                    w = np.max(lig_2d[:, 0]) - np.min(lig_2d[:, 0])
                    h = np.max(lig_2d[:, 1]) - np.min(lig_2d[:, 1])
                    target_scale = max(w, h, 4.0)
                    scale_map[dist_2d < 3.0] = target_scale

            slices.append({
                "input": np.stack([dist_ch, thick_ch, chem_ch], axis=0).astype(np.float32),
                "heatmap": heatmap,
                "scale": scale_map,
                "origin": origin.astype(np.float32), "u": u.astype(np.float32), 
                "v": v.astype(np.float32), "n": n.astype(np.float32), "resolution": float(resolution),
            })
    return slices

# 细化切片（评估时暂不用，但为了防报错保留）
def generate_refinement_slices(protein_coords, protein_chem, ligand_coords, candidate_center, radius=15.0, resolution=1.0):
    return generate_slices(protein_coords, protein_chem, ligand_coords, n_per_axis=5, resolution=resolution, half_size=radius, half_thick=radius)