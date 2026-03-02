#!/usr/bin/env python3
"""
数据驱动的 Skeleton 自动分块分析脚本
基于 HumanML3D 数据集 (nframe, 263) 的统计与相关性，自动将 skeleton 分块，
替代人为规定的 torso/arm/leg 划分。
"""
import os
import json
import argparse
from os.path import join as pjoin
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from tqdm import tqdm

# HumanML3D 263 维结构
# 0-3: root (4), 4-66: ric (63), 67-192: rot (126), 193-258: local_vel (66), 259-262: feet (4)
JOINTS_NUM = 22
FEAT_DIM = 263


def get_joint_dims(joint_id_1based):
    """
    获取指定关节对应的 263 维中的索引列表。
    joint_id_1based: 0=root, 1..21=body joints
    """
    if joint_id_1based == 0:  # root: root_data + root velocity
        return [0, 1, 2, 3, 193, 194, 195]
    i = joint_id_1based  # 1-based
    # ric: 4 + (i-1)*3, rot: 67 + (i-1)*6, vel: 193 + i*3
    return (
        [4 + (i - 1) * 3 + k for k in range(3)]
        + [4 + 63 + (i - 1) * 6 + k for k in range(6)]
        + [4 + 63 + 126 + i * 3 + k for k in range(3)]
    )


def get_feet_dims():
    """feet contact: 259-262"""
    return [259, 260, 261, 262]


def build_joint_to_dims():
    """构建 joint_id -> [dim_indices] 映射，joint_id 0..21"""
    mapping = {}
    mapping[0] = get_joint_dims(0)
    for j in range(1, 22):
        mapping[j] = get_joint_dims(j)
    return mapping


def load_motion_data(data_root, train_split_file, motion_dir, max_samples=None):
    """
    加载 HumanML3D 运动数据。
    返回: list of (T, 263) arrays
    """
    split_path = pjoin(data_root, train_split_file)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        id_list = [line.strip() for line in f.readlines() if line.strip()]

    motions = []
    for name in tqdm(id_list, desc="Loading motions"):
        npy_path = pjoin(motion_dir, name + ".npy")
        if not os.path.exists(npy_path):
            continue
        try:
            motion = np.load(npy_path)
            if motion.ndim != 2 or motion.shape[1] != FEAT_DIM:
                continue
            if motion.shape[0] < 10:  # 太短的序列跳过
                continue
            motions.append(motion)
            if max_samples is not None and len(motions) >= max_samples:
                break
        except Exception as e:
            continue

    return motions


def compute_joint_activation(motions, joint_to_dims):
    """
    对每个关节，提取其 12 维特征的每帧 L2 范数，得到 (N_total_frames,) 的激活序列。
    返回: dict joint_id -> activation_series (用于后续相关性计算)
    为节省内存，对每个样本计算后拼接，或使用采样。
    """
    joint_activations = {j: [] for j in range(22)}

    for motion in motions:
        T = motion.shape[0]
        for j in range(22):
            dims = joint_to_dims[j]
            feat = motion[:, dims]  # (T, n_dims)
            # 每帧 L2 范数
            act = np.linalg.norm(feat, axis=1)  # (T,)
            joint_activations[j].append(act)

    # 拼接所有样本
    for j in range(22):
        joint_activations[j] = np.concatenate(joint_activations[j], axis=0)

    return joint_activations


def compute_joint_correlation(joint_activations, max_frames=500000):
    """
    计算关节间相关性矩阵 (22 x 22)。
    若激活序列过长，可随机采样以控制内存。
    """
    n_joints = 22
    # 构建 (n_frames, 22) 矩阵
    total_len = min(len(joint_activations[0]), max_frames)
    if len(joint_activations[0]) > max_frames:
        idx = np.random.choice(len(joint_activations[0]), max_frames, replace=False)
        mat = np.stack([joint_activations[j][idx] for j in range(n_joints)], axis=1)
    else:
        mat = np.stack([joint_activations[j] for j in range(n_joints)], axis=1)

    # 相关系数矩阵
    corr = np.corrcoef(mat.T)  # (22, 22)
    np.nan_to_num(corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def hierarchical_cluster(corr_matrix, n_parts=6, method="average"):
    """
    基于相关性矩阵进行层次聚类。
    距离 = 1 - |correlation|，相关性高的关节距离近。
    """
    # 距离矩阵: 1 - |corr|
    dist = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0)

    # 转为 condensed form
    dist_condensed = squareform(dist, checks=False)

    # linkage
    Z = linkage(dist_condensed, method=method)

    # 切割得到 n_parts 个簇
    labels = fcluster(Z, n_parts, criterion="maxclust")
    return labels, Z


def part_seg_to_dim_indices(labels, joint_to_dims, feet_dims):
    """
    将聚类标签转为 partSeg 格式: list of lists of dim indices.
    labels: (22,) 每个关节的簇 id
    """
    unique_labels = np.unique(labels)
    part_seg = []
    part_to_joints = []  # 记录每个 part 包含的关节

    for lid in unique_labels:
        joint_ids = np.where(labels == lid)[0]
        dims = []
        for j in joint_ids:
            dims.extend(joint_to_dims[j])
        dims = sorted(set(dims))
        part_seg.append(dims)
        part_to_joints.append(set(joint_ids))

    # 将 feet [259,260,261,262] 并入已有 part，每个 dim 只归属一个 part
    # 259,260 -> 左腿; 261,262 -> 右腿
    left_leg_joints = {1, 4, 7, 10}
    right_leg_joints = {2, 5, 8, 11}
    feet_assigned = set()

    for p_idx, part in enumerate(part_seg):
        part_joints = part_to_joints[p_idx]
        # 左腿 part：加入 259,260（仅加入第一个含左腿的 part）
        if left_leg_joints & part_joints and 259 not in feet_assigned:
            for d in [259, 260]:
                part.append(d)
                feet_assigned.add(d)
        # 右腿 part：加入 261,262（仅加入第一个含右腿的 part）
        if right_leg_joints & part_joints and 261 not in feet_assigned:
            for d in [261, 262]:
                part.append(d)
                feet_assigned.add(d)

    # 若 feet 未被并入（如无单独腿簇），则并入 root 所在 part
    unassigned_feet = [d for d in feet_dims if d not in feet_assigned]
    if unassigned_feet:
        root_part_idx = None
        for p_idx, joints in enumerate(part_to_joints):
            if 0 in joints:  # root
                root_part_idx = p_idx
                break
        if root_part_idx is not None:
            for d in unassigned_feet:
                part_seg[root_part_idx].append(d)
        else:
            part_seg.append(unassigned_feet)

    # 排序每个 part 内的索引，并去重
    part_seg = [sorted(set(p)) for p in part_seg]
    return part_seg


def compute_per_dim_stats(motions):
    """计算每维的 mean, std, variance"""
    all_data = np.concatenate(motions, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    var = np.var(all_data, axis=0)
    return mean, std, var


def main():
    parser = argparse.ArgumentParser(description="Analyze HumanML3D skeleton and auto-partition")
    parser.add_argument("--data_root", type=str, default="./dataset/HumanML3D")
    parser.add_argument("--train_split", type=str, default="train.txt")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of motion samples") # None: all samples
    parser.add_argument("--n_parts", type=int, default=6)
    parser.add_argument("--method", type=str, default="average", choices=["ward", "average", "complete"])
    parser.add_argument("--output_dir", type=str, default="./output/partition_analysis")
    parser.add_argument("--max_frames", type=int, default=500000, help="Max frames for correlation (subsample if larger)")
    args = parser.parse_args()

    motion_dir = pjoin(args.data_root, "new_joint_vecs")
    train_split_path = pjoin(args.data_root, args.train_split)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据
    print("Loading motion data...")
    motions = load_motion_data(
        args.data_root,
        args.train_split,
        motion_dir,
        max_samples=args.max_samples,
    )
    if len(motions) == 0:
        raise RuntimeError("No valid motion data loaded. Check data_root and paths.")
    print(f"Loaded {len(motions)} motions")

    # 2. 关节-维度映射
    joint_to_dims = build_joint_to_dims()
    feet_dims = get_feet_dims()

    # 3. 每维统计
    print("Computing per-dimension statistics...")
    mean, std, var = compute_per_dim_stats(motions)
    np.savez(
        pjoin(args.output_dir, "skeleton_stats.npz"),
        mean=mean,
        std=std,
        var=var,
    )

    # 4. 关节激活与相关性
    print("Computing joint activations and correlations...")
    joint_activations = compute_joint_activation(motions, joint_to_dims)
    corr_matrix = compute_joint_correlation(joint_activations, max_frames=args.max_frames)
    np.savez(
        pjoin(args.output_dir, "skeleton_stats.npz"),
        mean=mean,
        std=std,
        var=var,
        corr_matrix=corr_matrix,
    )

    # 5. 层次聚类
    print(f"Clustering into {args.n_parts} parts (method={args.method})...")
    labels, Z = hierarchical_cluster(corr_matrix, n_parts=args.n_parts, method=args.method)

    # 6. 转为 partSeg
    part_seg = part_seg_to_dim_indices(labels, joint_to_dims, feet_dims)

    # 7. 保存 part_seg
    result = {
        "partSeg": part_seg,
        "n_parts": len(part_seg),
        "joint_labels": labels.tolist(),
        "cluster_method": args.method,
        "n_samples": len(motions),
    }
    with open(pjoin(args.output_dir, "skeleton_partition2.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 8. 树状图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=[f"J{i}" for i in range(22)])
        plt.title("Hierarchical Clustering of Joints (by correlation)")
        plt.xlabel("Joint")
        plt.tight_layout()
        plt.savefig(pjoin(args.output_dir, "dendrogram.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Could not save dendrogram: {e}")

    # 9. 文本报告
    report_lines = [
        "=== Skeleton Partition Analysis Report ===",
        f"Data root: {args.data_root}",
        f"# motions: {len(motions)}",
        f"# parts: {len(part_seg)}",
        f"Cluster method: {args.method}",
        "",
        "Part segmentation (dim indices):",
    ]
    for i, part in enumerate(part_seg):
        report_lines.append(f"  Part {i}: {len(part)} dims -> {part[:20]}{'...' if len(part) > 20 else ''}")
    report_lines.append("")
    report_lines.append("Joint cluster labels:")
    for j in range(22):
        report_lines.append(f"  Joint {j}: cluster {labels[j]}")

    report_path = pjoin(args.output_dir, "partition_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to {report_path}")
    print(f"Partition saved to {pjoin(args.output_dir, 'skeleton_partition2.json')}")


if __name__ == "__main__":
    main()
