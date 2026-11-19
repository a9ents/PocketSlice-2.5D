import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TinyUNet

torch.manual_seed(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache_slices")


class CachedSliceDataset(Dataset):
    def __init__(self, cache_dir, pdb_ids, is_train=False):
        self.files = [os.path.join(cache_dir, f"{pid}.pt") for pid in pdb_ids
                      if os.path.exists(os.path.join(cache_dir, f"{pid}.pt"))]
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        inputs = data["inputs"]
        heatmaps = data["heatmaps"]
        scales = data["scales"]

        # val：返回全部切片，由训练循环展开
        if not self.is_train:
            return inputs, heatmaps, scales

        # train：优先采样含口袋切片，负片比例 ~50%
        positive_idx = torch.where(heatmaps.view(heatmaps.shape[0], -1).max(dim=1)[0] > 0.5)[0]
        if len(positive_idx) > 0 and torch.rand(1) > 0.5:
            i = positive_idx[torch.randint(0, len(positive_idx), (1,)).item()].item()
        else:
            i = torch.randint(0, inputs.shape[0], (1,)).item()

        x, y_heat, y_scale = inputs[i], heatmaps[i], scales[i]

        # 数据增强：x 与标签同步变换
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
            y_heat = torch.flip(y_heat, dims=[0])
            y_scale = torch.flip(y_scale, dims=[0])
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
            y_heat = torch.flip(y_heat, dims=[1])
            y_scale = torch.flip(y_scale, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
            y_heat = torch.rot90(y_heat, k, dims=[0, 1])
            y_scale = torch.rot90(y_scale, k, dims=[0, 1])

        return x, y_heat, y_scale


def center_focal_loss(pred_heatmap, gt_heatmap, pred_scale, gt_scale):
    """CenterNet focal loss + L1 scale loss。
    注意：pred_heatmap / gt_heatmap / pred_scale / gt_scale 形状均为 (B, H, W)。
    """
    mask = gt_heatmap >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_heatmap.device, requires_grad=True)

    # 全部用同一个 mask 展平成 1D，避免 (M,) 和 (B,H,W) 混用
    pred       = torch.clamp(pred_heatmap[mask], 1e-4, 1 - 1e-4)   # (M,)
    gt         = gt_heatmap[mask]                                   # (M,)
    pred_s     = pred_scale[mask]                                   # (M,)
    gt_s       = gt_scale[mask]                                     # (M,)

    pos_mask = gt > 0.9
    neg_mask = gt < 0.1

    pos_loss = (-torch.log(pred[pos_mask]) * ((1 - pred[pos_mask]) ** 2)).sum()
    neg_loss = (-torch.log(1 - pred[neg_mask])
                * (pred[neg_mask] ** 2)
                * ((1 - gt[neg_mask]) ** 4)).sum()

    num_pos = pos_mask.float().sum().clamp(min=1.0)   # 防止纯负片 batch
    heatmap_loss = (pos_loss + neg_loss) / num_pos

    # scale 只在"正样本 且 gt_scale > 0"处监督；两边都已展平
    scale_mask = pos_mask & (gt_s > 0)
    if scale_mask.sum() > 0:
        scale_loss = F.l1_loss(pred_s[scale_mask], gt_s[scale_mask])
    else:
        scale_loss = torch.tensor(0.0, device=pred_heatmap.device)

    return heatmap_loss + 0.1 * scale_loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = torch.load("dataset_split.pt")
    train_ids = list(split["train"].keys())

    val_idx = int(len(train_ids) * 0.9)
    train_loader = DataLoader(
        CachedSliceDataset(CACHE_DIR, train_ids[:val_idx], True),
        batch_size=16, shuffle=True, num_workers=0,
    )
    # val 每个蛋白返回全切片，因此 batch_size=1
    val_loader = DataLoader(
        CachedSliceDataset(CACHE_DIR, train_ids[val_idx:], False),
        batch_size=1, shuffle=False, num_workers=0,
    )

    model = TinyUNet(in_ch=3, out_ch=2, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')

    for ep in range(1, 101):
        # ---------- Train ----------
        model.train()
        tr_loss, tr_n = 0.0, 0
        for x, y_heat, y_scale in tqdm(train_loader, desc=f"Epoch {ep} [Train]", leave=False):
            x, y_heat, y_scale = x.to(device), y_heat.to(device), y_scale.to(device)
            pred_heat, pred_scale = model(x)
            loss = center_focal_loss(pred_heat[:, 0], y_heat, pred_scale[:, 0], y_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            tr_n += 1
        tr_loss /= max(tr_n, 1)

        # ---------- Val ----------
        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for xs, yhs, yss in val_loader:
                xs = xs.squeeze(0).to(device)    # (N, C, H, W)
                yhs = yhs.squeeze(0).to(device)  # (N, H, W)
                yss = yss.squeeze(0).to(device)
                ph, ps = model(xs)               # (N, 2, H, W)
                va_loss += center_focal_loss(ph[:, 0], yhs, ps[:, 0], yss).item()
                va_n += 1
        va_loss /= max(va_n, 1)

        print(f"Epoch {ep:3d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "tiny_unet_centernet.pt"))
            print("  -> 模型已保存 (Best)")


if __name__ == "__main__":
    train()