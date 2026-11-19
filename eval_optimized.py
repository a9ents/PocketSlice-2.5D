import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import glob
import json
import argparse
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from pdb_utils import load_pdb, ligand_centroid
from slicing import generate_slices
from model import TinyUNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = os.path.join(BASE_DIR, "coach420_pdbs")
MODEL_PATH = os.path.join(BASE_DIR, "tiny_unet_centernet.pt")
PREDS_OUT = os.path.join(BASE_DIR, "preds_ours.json")
HIT_DIST = 4.0


def parse_chain_id(pdb_path):
    """取第一个单字母、非数字的段；兼容 1abc.pdb / 1abc_A.pdb / 1abc_A_ATP.pdb。"""
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    parts = re.split(r"[_\-]", stem)
    for p in parts[1:]:
        if len(p) == 1 and p.isalpha():
            return p
    return None


def extract_peaks(slices, model, device, threshold=0.2,
                  use_scale_filter=False, scale_lo=2.0, scale_hi=30.0,
                  zero_channels=None):
    """每片跑模型 → 局部最大值 → 反投影到 3D → 多轴投票 + NMS聚合。"""
    candidates = []
    model.eval()
    
    for s in slices:
        x = torch.from_numpy(s["input"]).unsqueeze(0).to(device)
        if zero_channels:
            for ch in zero_channels:
                x[:, ch, :, :] = 0.0
                
        with torch.no_grad():
            heatmap, scale_map = model(x)
        heatmap = heatmap[0, 0].cpu().numpy()
        scale_map = scale_map[0, 0].cpu().numpy()

        # 局部极大值检测
        local_max = ndimage.maximum_filter(heatmap, size=5) >= (heatmap - 1e-6)
        peaks = local_max & (heatmap > threshold)
        ys, xs = np.where(peaks)

        # 【新增优化 1】计算 3x3 邻域平均置信度
        mean_map = ndimage.uniform_filter(heatmap, size=3)

        for y, x_px in zip(ys, xs):
            raw_score = float(heatmap[y, x_px])
            local_mean = float(mean_map[y, x_px])
            pred_scale = float(scale_map[y, x_px])

            # 如果开启了硬过滤，直接跳过
            if use_scale_filter:
                if pred_scale < scale_lo or pred_scale > scale_hi:
                    continue

            # 【新增优化 2】置信度平滑
            # 【新增优化 2】置信度平滑 (减弱)
            smooth_score = 0.8 * raw_score + 0.2 * local_mean
            
            # 【新增优化 3】尺寸软惩罚 (放宽)
            scale_penalty = 1.0 if pred_scale <= 20.0 else (20.0 / pred_scale)
            final_score = smooth_score * scale_penalty

            phys = s["origin"] + x_px * s["u"] + y * s["v"]
            candidates.append({
                "center": phys.astype(np.float32),
                "score": final_score,       # 使用优化后的分数
                "raw_score": raw_score,     # 保留原始分数用于 debug
                "normal": s["n"].astype(np.float32),
                "scale": pred_scale
            })

    # 多轴投票与空间 NMS（非极大值抑制）
    candidates.sort(key=lambda c: -c["score"])
    unique_cands = []
    
    for c in candidates:
        merged = False
        for uc in unique_cands:
            axis_differ = abs(float(np.dot(c["normal"], uc["normal"]))) < 0.7
            dist = np.linalg.norm(c["center"] - uc["center"])
            
            # 情况 A：多轴投票（来自不同切片轴，且距离 < 3.0 Å）
            if axis_differ and dist < 3.0:
                uc["votes"] += 1
                # 如果当前点得分更高，更新中心点和分数
                if c["score"] > uc["score"]:
                    uc["center"] = c["center"]
                    uc["score"] = c["score"]
                    uc["scale"] = c["scale"]
                merged = True
                break
                
            # 【新增优化 4】情况 B：同轴/异轴空间抑制（距离 < 6.0 Å 强制合并，防止一个口袋碎片占多个名额）
            # 【新增优化 4】情况 B：同轴/异轴空间抑制 (缩小半径)
            elif dist < 4.0:
                if c["score"] > uc["score"]:
                    uc["center"] = c["center"]
                    uc["score"] = c["score"]
                merged = True
                break

        if not merged:
            unique_cands.append({**c, "votes": 1})

    # 最终排序：先按票数(votes)降序，再按优化后的分数(score)降序
    unique_cands.sort(key=lambda c: (-c["votes"], -c["score"]))
    return unique_cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--scale-filter", action="store_true",
                        help="开启 scale 过滤（默认关）")
    parser.add_argument("--scale-lo", type=float, default=2.0)
    parser.add_argument("--scale-hi", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个蛋白，用于快速验证")
    parser.add_argument("--no-dump", action="store_true",
                        help="不导出 preds_ours.json（消融时用）")
    parser.add_argument("--zero-channels", type=str, default=None,
                        help="逗号分隔的通道索引置零。0=dist, 1=thick, 2=chem。"
                             "如 '1' 或 '1,2' 或 '0,1,2'")
    args = parser.parse_args()

    # 解析 zero_channels
    zero_ch = None
    if args.zero_channels:
        zero_ch = [int(c) for c in args.zero_channels.split(",")]
        for c in zero_ch:
            if c not in (0, 1, 2):
                raise ValueError(f"通道索引必须是 0/1/2，收到 {c}")

    device = "cpu"
    model = TinyUNet(in_ch=3, out_ch=2, base=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    split = torch.load(os.path.join(BASE_DIR, "dataset_split.pt"), map_location="cpu")
    test_items = list(split["test"].items())
    if args.limit:
        test_items = test_items[:args.limit]

    zc_str = args.zero_channels if args.zero_channels else "无"
    print(f"盲测配置: threshold={args.threshold}  "
          f"scale_filter={'ON' if args.scale_filter else 'OFF'}  "
          f"zero_channels={zc_str} (0=dist,1=thick,2=chem)  "
          f"蛋白数={len(test_items)}\n")

    results = []
    preds_dump = {}
    skipped = 0
    empty_cand = 0
    error_examples = []

    for pdb_id, lig_name in tqdm(test_items, desc="Testing"):
        matches = (glob.glob(os.path.join(PDB_DIR, f"{pdb_id}*.pdb"))
                   or glob.glob(os.path.join(PDB_DIR, f"{pdb_id.upper()}*.pdb")))
        if not matches:
            skipped += 1
            continue

        try:
            pdb_path = matches[0]
            chain_id = parse_chain_id(pdb_path)
            prot, prot_chem, lig = load_pdb(pdb_path, lig_name, chain_id=chain_id)
            if len(prot) < 50 or len(lig) == 0:
                skipped += 1
                continue
            lig_c = ligand_centroid(lig)

            slices = generate_slices(prot, prot_chem, lig, n_per_axis=25)
            cands = extract_peaks(
                slices, model, device,
                threshold=args.threshold,
                use_scale_filter=args.scale_filter,
                scale_lo=args.scale_lo,
                scale_hi=args.scale_hi,
                zero_channels=zero_ch,
            )

            # 导出预测（仅基准跑，消融用 --no-dump 关闭）
            if not args.no_dump:
                preds_dump[pdb_id] = [
                    [float(c["center"][0]), float(c["center"][1]),
                     float(c["center"][2]), float(c["score"])]
                    for c in cands
                ]

            if len(cands) == 0:
                empty_cand += 1
                tqdm.write(f"  {pdb_id}: 0 候选")
                results.append({"hit": False, "min_dist_all": None,
                                "min_dist_top10": None, "n_cands": 0})
                continue

            all_dists = [float(np.linalg.norm(c["center"] - lig_c)) for c in cands]
            top10_dists = all_dists[:10]
            hit = any(d <= HIT_DIST for d in top10_dists)
            
            # 增加 Top-1 和 Top-3 的命中统计
            hit1 = all_dists[0] <= HIT_DIST
            hit3 = any(d <= HIT_DIST for d in all_dists[:3])

            results.append({
                "hit": hit,
                "hit1": hit1,
                "hit3": hit3,
                "min_dist_all": min(all_dists),
                "min_dist_top10": min(top10_dists),
                "n_cands": len(cands),
            })

            nearest_rank = int(np.argmin(all_dists)) + 1
            top3_scores = [round(c["score"], 3) for c in cands[:3]]
            tqdm.write(
                f"  {pdb_id}: cands={len(cands):3d}  "
                f"min_all={min(all_dists):5.2f}A(rank {nearest_rank})  "
                f"min_top10={min(top10_dists):5.2f}A  "
                f"top3score={top3_scores}  hit={'Y' if hit else 'N'}"
            )

        except Exception as e:
            skipped += 1
            if len(error_examples) < 5:
                error_examples.append((pdb_id, str(e)))
            continue

    # ---------- 导出预测 JSON（仅基准跑） ----------
    if not args.no_dump:
        json.dump(preds_dump, open(PREDS_OUT, "w", encoding="utf-8"), indent=1)
        print(f"\n预测已导出: {PREDS_OUT} ({len(preds_dump)} 个蛋白)")

    # ---------- 汇总 ----------
    total = len(results)
    valid = [r for r in results if r["min_dist_all"] is not None]
    hits = sum(r["hit"] for r in results)
    
    # 统计 Top-1 和 Top-3
    hits1 = sum(r.get("hit1", False) for r in results)
    hits3 = sum(r.get("hit3", False) for r in results)

    if valid:
        mean_min_all = float(np.mean([r["min_dist_all"] for r in valid]))
        within4_any = sum(1 for r in valid if r["min_dist_all"] <= HIT_DIST)
        rank_fail = sum(1 for r in valid
                        if r["min_dist_all"] <= HIT_DIST and not r["hit"])
    else:
        mean_min_all = float("nan")
        within4_any = 0
        rank_fail = 0

    print("\n" + "=" * 60)
    print("===== CenterNet 盲测结果 =====")
    print(f"threshold        : {args.threshold}")
    print(f"scale 过滤       : "
          f"{'ON ' + str((args.scale_lo, args.scale_hi)) if args.scale_filter else 'OFF'}")
    print(f"置零通道         : {zc_str}  (0=dist, 1=thick, 2=chem)")
    print(f"输入蛋白数       : {len(test_items)}")
    print(f"跳过异常数       : {skipped}")
    print(f"成功评估数       : {total}")
    print(f"空候选蛋白数     : {empty_cand}")
    
    # 打印新增的 Top-N 指标
    print(f"Top-1 命中数     : {hits1}")
    print(f"Top-3 命中数     : {hits3}")
    print(f"命中数 (Top-10)  : {hits}")
    
    print(f"Top-1 准确率     : {hits1 / max(total, 1) * 100:.1f}%")
    print(f"Top-3 准确率     : {hits3 / max(total, 1) * 100:.1f}%")
    print(f"Oracle@10        : {hits / max(total, 1):.3f}  "
          f"({hits / max(total, 1) * 100:.1f}%)")
    
    print("-" * 60)
    print("诊断（仅统计有候选的蛋白）:")
    print(f"  平均最近候选距配体 : {mean_min_all:.2f} A")
    print(f"  有候选进入 4A 的蛋白: {within4_any} / {len(valid)}")
    print(f"  排序失败数         : {rank_fail}   "
          f"(有 4A 内候选但没进 top10)")
    if error_examples:
        print("-" * 60)
        print("异常示例（前 5 条）:")
        for pid, msg in error_examples:
            print(f"  {pid}: {msg}")
    print("=" * 60)


if __name__ == "__main__":
    main()