#!/usr/bin/env python3
"""
Control-oriented skeleton partition analysis for HumanML3D (263-dim vectors).

Key differences vs old script:
1) uses relative-to-parent motion activations to reduce root/global-motion bias,
2) uses lagged absolute correlation for phase-shifted coupling,
3) applies kinematic chain post-processing constraints for control stability,
4) optionally keeps feet contact dims as a standalone part.
"""

import argparse
import json
import os
from collections import defaultdict
from os.path import join as pjoin

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from tqdm import tqdm


JOINTS_NUM = 22
FEAT_DIM = 263

# HumanML3D/t2m kinematic chains (joint ids in [0, 21])
T2M_CHAINS = {
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "spine": [3, 6, 9, 12, 15],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
}

# Minimal hard constraints for control-oriented partitioning
CONSTRAINT_CHAINS = {
    "left_leg": T2M_CHAINS["left_leg"],
    "right_leg": T2M_CHAINS["right_leg"],
    "left_arm": T2M_CHAINS["left_arm"],
    "right_arm": T2M_CHAINS["right_arm"],
}

JOINT_NAMES = [
    "root/pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def get_joint_dims(joint_id):
    """Return dim indices for one joint in HumanML3D 263-dim vector."""
    if joint_id == 0:
        # root orientation/velocity block + root local vel block
        return {
            "all": [0, 1, 2, 3, 193, 194, 195],
            "ric": [0, 1, 2, 3],
            "rot": [],
            "vel": [193, 194, 195],
        }

    i = joint_id
    ric = [4 + (i - 1) * 3 + k for k in range(3)]
    rot = [67 + (i - 1) * 6 + k for k in range(6)]
    vel = [193 + i * 3 + k for k in range(3)]
    return {
        "all": ric + rot + vel,
        "ric": ric,
        "rot": rot,
        "vel": vel,
    }


def build_joint_to_dims():
    return {j: get_joint_dims(j) for j in range(JOINTS_NUM)}


def get_feet_dims():
    return [259, 260, 261, 262]


def build_parent_map():
    """Build parent index map for t2m 22-joint skeleton."""
    parent = {0: -1}
    all_chains = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
    for chain in all_chains:
        for p, c in zip(chain[:-1], chain[1:]):
            parent[c] = p

    missing = [j for j in range(JOINTS_NUM) if j not in parent]
    if missing:
        raise RuntimeError(f"Parent map missing joints: {missing}")
    return parent


def load_motion_data(data_root, train_split_file, motion_dir, max_samples=None):
    split_path = pjoin(data_root, train_split_file)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        id_list = [line.strip() for line in f if line.strip()]

    motions = []
    skipped_missing = 0
    skipped_shape = 0

    for name in tqdm(id_list, desc="Loading motions"):
        npy_path = pjoin(motion_dir, name + ".npy")
        if not os.path.exists(npy_path):
            skipped_missing += 1
            continue

        try:
            motion = np.load(npy_path)
        except Exception:
            skipped_shape += 1
            continue

        if motion.ndim != 2 or motion.shape[1] != FEAT_DIM or motion.shape[0] < 10:
            skipped_shape += 1
            continue

        motions.append(motion)
        if max_samples is not None and len(motions) >= max_samples:
            break

    if not motions:
        raise RuntimeError("No valid motion data loaded. Check data_root/train_split/motion_dir.")

    print(
        f"Loaded motions: {len(motions)} | skipped_missing: {skipped_missing} | skipped_invalid: {skipped_shape}"
    )
    return motions


def compute_per_dim_stats(motions):
    all_data = np.concatenate(motions, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    var = np.var(all_data, axis=0)
    return mean, std, var


def _joint_feature_sequence(motion, joint_id, joint_to_dims, parent_map, mode):
    """
    Build per-frame feature sequence used for activation.

    mode='absolute': use this joint's native dims.
    mode='relative_parent': use child-parent deltas for ric/rot/vel blocks.
    """
    dims = joint_to_dims[joint_id]

    if mode == "absolute":
        return motion[:, dims["all"]]

    if mode != "relative_parent":
        raise ValueError(f"Unsupported feature mode: {mode}")

    if joint_id == 0:
        return motion[:, dims["all"]]

    p = parent_map[joint_id]
    child_ric = motion[:, dims["ric"]]
    child_rot = motion[:, dims["rot"]]
    child_vel = motion[:, dims["vel"]]

    if p <= 0:
        # parent is root: ric is already root-relative, keep as-is.
        rel_ric = child_ric

        if p == 0:
            p_vel = motion[:, joint_to_dims[0]["vel"]]
            rel_vel = child_vel - p_vel
        else:
            rel_vel = child_vel

        rel_rot = child_rot
    else:
        p_dims = joint_to_dims[p]
        p_ric = motion[:, p_dims["ric"]]
        p_rot = motion[:, p_dims["rot"]]
        p_vel = motion[:, p_dims["vel"]]

        rel_ric = child_ric - p_ric
        rel_rot = child_rot - p_rot
        rel_vel = child_vel - p_vel

    return np.concatenate([rel_ric, rel_rot, rel_vel], axis=1)


def compute_joint_activations(motions, joint_to_dims, parent_map, feature_mode="relative_parent"):
    """
    Return activation matrix shape (N_total_frames, 22), each column a joint activation.
    activation = L2 norm of selected per-frame feature vector.
    """
    all_acts = []

    for motion in tqdm(motions, desc="Computing joint activations"):
        T = motion.shape[0]
        acts = np.empty((T, JOINTS_NUM), dtype=np.float32)

        for j in range(JOINTS_NUM):
            feat = _joint_feature_sequence(
                motion=motion,
                joint_id=j,
                joint_to_dims=joint_to_dims,
                parent_map=parent_map,
                mode=feature_mode,
            )
            acts[:, j] = np.linalg.norm(feat, axis=1)

        all_acts.append(acts)

    activation_mat = np.concatenate(all_acts, axis=0)
    return activation_mat


def maybe_subsample_frames(mat, max_frames, seed):
    if mat.shape[0] <= max_frames:
        return mat

    rng = np.random.default_rng(seed)
    idx = rng.choice(mat.shape[0], size=max_frames, replace=False)
    idx.sort()
    return mat[idx]


def compute_similarity_matrix(activation_mat, max_lag=4, eps=1e-8):
    """
    Compute lag-aware absolute correlation similarity.

    sim[i, j] = max_{tau in [-max_lag, max_lag]} |corr(a_i[t], a_j[t+tau])|
    """
    n_frames, n_joints = activation_mat.shape
    if n_joints != JOINTS_NUM:
        raise RuntimeError(f"Expected {JOINTS_NUM} joints, got {n_joints}")

    x = activation_mat.astype(np.float32, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    z = (x - mean) / np.maximum(std, eps)

    sim = np.eye(n_joints, dtype=np.float32)

    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            best = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    a = z[:-lag, i]
                    b = z[lag:, j]
                elif lag < 0:
                    a = z[-lag:, i]
                    b = z[:lag, j]
                else:
                    a = z[:, i]
                    b = z[:, j]

                if a.size < 2:
                    continue

                c = float(np.mean(a * b))
                c = abs(c)
                if c > best:
                    best = c

            sim[i, j] = best
            sim[j, i] = best

    np.nan_to_num(sim, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def hierarchical_cluster(similarity, n_parts=6, method="average"):
    dist = 1.0 - similarity
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method=method)
    labels = fcluster(Z, n_parts, criterion="maxclust")
    return labels, Z


def enforce_chain_constraints(labels, chain_defs):
    """
    Force each chain's joints to share one label.
    Use distal joint's label as the target label to bias toward control endpoint semantics.
    """
    out = labels.copy()
    changes = []

    for chain_name, chain in chain_defs.items():
        old_chain_labels = [int(out[j]) for j in chain]
        target = int(out[chain[-1]])
        for j in chain:
            out[j] = target
        if len(set(old_chain_labels)) > 1:
            changes.append(
                {
                    "chain": chain_name,
                    "joints": chain,
                    "old_labels": old_chain_labels,
                    "new_label": target,
                }
            )

    return out, changes


def remap_labels_contiguous(labels):
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: i for i, old in enumerate(uniq)}
    new_labels = np.array([remap[int(x)] for x in labels], dtype=np.int32)
    return new_labels, remap


def split_largest_cluster(labels, similarity, protected_joints=None):
    """Split the largest cluster into two sub-clusters using hierarchical clustering."""
    if protected_joints is None:
        protected_joints = set()
    else:
        protected_joints = set(protected_joints)

    uniq, counts = np.unique(labels, return_counts=True)
    # Prefer splitting clusters that have enough non-protected joints.
    label_order = [int(x) for x in uniq[np.argsort(-counts)]]
    largest_label = None
    members = None
    split_candidates = None

    for lab in label_order:
        m = np.where(labels == lab)[0]
        cands = [j for j in m.tolist() if j not in protected_joints]
        if len(cands) >= 2:
            largest_label = lab
            members = m
            split_candidates = np.array(cands, dtype=np.int32)
            break

    if largest_label is None:
        return labels, False

    sub_sim = similarity[np.ix_(split_candidates, split_candidates)]
    sub_dist = 1.0 - sub_sim
    np.fill_diagonal(sub_dist, 0.0)

    sub_cond = squareform(sub_dist, checks=False)
    sub_z = linkage(sub_cond, method="average")
    sub_labels = fcluster(sub_z, 2, criterion="maxclust")

    new_label = int(labels.max()) + 1
    out = labels.copy()
    for idx, joint_idx in enumerate(split_candidates):
        if sub_labels[idx] == 2:
            out[joint_idx] = new_label

    return out, True


def ensure_joint_cluster_count(labels, target_joint_parts, similarity, protected_joints=None):
    """Adjust labels so number of joint clusters equals target_joint_parts."""
    out = labels.copy()

    while len(np.unique(out)) < target_joint_parts:
        out, ok = split_largest_cluster(
            labels=out,
            similarity=similarity,
            protected_joints=protected_joints,
        )
        if not ok:
            break

    while len(np.unique(out)) > target_joint_parts:
        uniq, counts = np.unique(out, return_counts=True)
        smallest = int(uniq[np.argmin(counts)])
        if len(uniq) <= 1:
            break

        smallest_members = np.where(out == smallest)[0]
        candidates = [int(u) for u in uniq if int(u) != smallest]

        best_target = None
        best_score = -1.0
        for c in candidates:
            c_members = np.where(out == c)[0]
            score = float(similarity[np.ix_(smallest_members, c_members)].mean())
            if score > best_score:
                best_score = score
                best_target = c

        out[smallest_members] = best_target

    return out


def labels_to_part_seg(labels, joint_to_dims, add_contact_part=True):
    """
    Build partSeg dims from joint labels.
    If add_contact_part=True, feet contact dims [259:262] become a standalone part.
    """
    part_dims = defaultdict(list)
    part_joints = defaultdict(list)

    for j in range(JOINTS_NUM):
        lab = int(labels[j])
        part_dims[lab].extend(joint_to_dims[j]["all"])
        part_joints[lab].append(j)

    ordered_labels = sorted(part_dims.keys())
    part_seg = [sorted(set(part_dims[lab])) for lab in ordered_labels]
    part_joint_list = [sorted(part_joints[lab]) for lab in ordered_labels]

    if add_contact_part:
        part_seg.append(get_feet_dims())
        part_joint_list.append([])
    else:
        # attach feet dims to leg parts (left foot contact -> left leg, right -> right leg)
        left_label = int(labels[T2M_CHAINS["left_leg"][-1]])
        right_label = int(labels[T2M_CHAINS["right_leg"][-1]])
        left_idx = ordered_labels.index(left_label)
        right_idx = ordered_labels.index(right_label)

        part_seg[left_idx] = sorted(set(part_seg[left_idx] + [259, 260]))
        part_seg[right_idx] = sorted(set(part_seg[right_idx] + [261, 262]))

    return part_seg, part_joint_list


def save_dendrogram(Z, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=[f"J{i}" for i in range(JOINTS_NUM)])
        plt.title("Joint Clustering Dendrogram (lagged relative-motion similarity)")
        plt.xlabel("Joint")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception as exc:
        print(f"[warn] Failed to save dendrogram: {exc}")


def save_leg_activation_plot(motions, joint_to_dims, parent_map, feature_mode, out_path):
    if not motions:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip activation plot: {exc}")
        return

    # Pick one medium-length sample for readability.
    idx = int(np.argmin(np.abs(np.array([m.shape[0] for m in motions]) - 160)))
    motion = motions[idx]

    left_chain = T2M_CHAINS["left_leg"]
    right_chain = T2M_CHAINS["right_leg"]

    def chain_acts(chain):
        out = []
        for j in chain:
            feat = _joint_feature_sequence(motion, j, joint_to_dims, parent_map, feature_mode)
            out.append(np.linalg.norm(feat, axis=1))
        return out

    left_acts = chain_acts(left_chain)
    right_acts = chain_acts(right_chain)

    t = np.arange(motion.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    left_names = [JOINT_NAMES[j] for j in left_chain]
    right_names = [JOINT_NAMES[j] for j in right_chain]

    for y, name in zip(left_acts, left_names):
        axes[0].plot(t, y, label=name, linewidth=1.2)
    axes[0].set_title("Left leg chain activations")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[0].grid(alpha=0.2)

    for y, name in zip(right_acts, right_names):
        axes[1].plot(t, y, label=name, linewidth=1.2)
    axes[1].set_title("Right leg chain activations")
    axes[1].legend(loc="upper right", ncol=2, fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[1].set_xlabel("Frame")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def build_report(
    args,
    motions,
    labels,
    part_seg,
    part_joint_list,
    post_changes,
    similarity,
    output_dir,
):
    lines = []
    lines.append("=== Control-Oriented Skeleton Partition Report ===")
    lines.append(f"Data root: {args.data_root}")
    lines.append(f"Motions loaded: {len(motions)}")
    lines.append(f"Feature mode: {args.feature_mode}")
    lines.append(f"Lagged correlation max_lag: {args.max_lag}")
    lines.append(f"Initial cluster method: {args.method}")
    lines.append(f"Target n_parts: {args.n_parts}")
    lines.append(f"Standalone contact part: {args.add_contact_part}")
    lines.append("")

    lines.append("Post-processing chain fixes:")
    if not post_changes:
        lines.append("  (none)")
    else:
        for c in post_changes:
            lines.append(
                f"  - {c['chain']}: joints {c['joints']} labels {c['old_labels']} -> {c['new_label']}"
            )

    lines.append("")
    lines.append("Joint labels:")
    for j in range(JOINTS_NUM):
        lines.append(f"  J{j:02d} {JOINT_NAMES[j]:>14s} -> part {int(labels[j])}")

    lines.append("")
    lines.append("Part summary:")
    for i, (dims, joints) in enumerate(zip(part_seg, part_joint_list)):
        if joints:
            j_desc = ", ".join([f"J{j}({JOINT_NAMES[j]})" for j in joints])
        else:
            j_desc = "[contact-only dims]"
        lines.append(f"  Part {i}: {len(dims)} dims | joints: {j_desc}")

    lines.append("")
    lines.append("Selected pair similarities (for sanity):")
    key_pairs = [
        (1, 4), (4, 7), (7, 10),
        (2, 5), (5, 8), (8, 11),
        (13, 16), (16, 18), (18, 20),
        (14, 17), (17, 19), (19, 21),
    ]
    for a, b in key_pairs:
        lines.append(
            f"  sim(J{a:02d},{b:02d}) = {float(similarity[a, b]):.4f}"
        )

    report_path = pjoin(output_dir, "partition_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Control-oriented HumanML3D skeleton partition analysis")
    parser.add_argument("--data_root", type=str, default="./dataset/HumanML3D")
    parser.add_argument("--train_split", type=str, default="train.txt")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--n_parts", type=int, default=6)
    parser.add_argument("--method", type=str, default="average", choices=["average", "complete", "ward"])
    parser.add_argument("--feature_mode", type=str, default="relative_parent", choices=["absolute", "relative_parent"])
    parser.add_argument("--max_lag", type=int, default=4)

    parser.add_argument("--enforce_chain_constraints", action="store_true", default=True)
    parser.add_argument("--no_enforce_chain_constraints", dest="enforce_chain_constraints", action="store_false")

    parser.add_argument("--add_contact_part", action="store_true", default=True)
    parser.add_argument("--no_add_contact_part", dest="add_contact_part", action="store_false")

    parser.add_argument("--sync_primary_partition", action="store_true", default=False,
                        help="Also write result to skeleton_partition.json in output_dir")

    parser.add_argument("--output_dir", type=str, default="./partition_analysis")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    motion_dir = pjoin(args.data_root, "new_joint_vecs")

    print("[1/7] Loading motion data...")
    motions = load_motion_data(
        data_root=args.data_root,
        train_split_file=args.train_split,
        motion_dir=motion_dir,
        max_samples=args.max_samples,
    )

    print("[2/7] Building joint metadata...")
    joint_to_dims = build_joint_to_dims()
    parent_map = build_parent_map()

    print("[3/7] Computing per-dim stats...")
    mean, std, var = compute_per_dim_stats(motions)

    print("[4/7] Computing activations...")
    activation_mat = compute_joint_activations(
        motions=motions,
        joint_to_dims=joint_to_dims,
        parent_map=parent_map,
        feature_mode=args.feature_mode,
    )

    activation_sample = maybe_subsample_frames(activation_mat, args.max_frames, args.seed)
    print(f"Activation frames: full={activation_mat.shape[0]} sampled={activation_sample.shape[0]}")

    print("[5/7] Computing lagged similarity + clustering...")
    similarity = compute_similarity_matrix(activation_sample, max_lag=args.max_lag)
    labels_raw, Z = hierarchical_cluster(similarity, n_parts=args.n_parts, method=args.method)

    labels = labels_raw.copy()
    post_changes = []

    desired_joint_parts = args.n_parts - 1 if args.add_contact_part else args.n_parts
    desired_joint_parts = max(1, desired_joint_parts)
    protected_joints = sorted(
        set(j for chain in CONSTRAINT_CHAINS.values() for j in chain)
    )

    if args.enforce_chain_constraints:
        labels, post_changes = enforce_chain_constraints(labels, CONSTRAINT_CHAINS)

    labels = ensure_joint_cluster_count(
        labels=labels,
        target_joint_parts=desired_joint_parts,
        similarity=similarity,
        protected_joints=protected_joints,
    )
    # Final safety pass: keep constrained chains intact after cluster-count balancing.
    if args.enforce_chain_constraints:
        labels, _ = enforce_chain_constraints(labels, CONSTRAINT_CHAINS)
    labels, _ = remap_labels_contiguous(labels)

    print("[6/7] Building partSeg and saving outputs...")
    part_seg, part_joint_list = labels_to_part_seg(
        labels=labels,
        joint_to_dims=joint_to_dims,
        add_contact_part=args.add_contact_part,
    )

    result = {
        "partSeg": part_seg,
        "n_parts": len(part_seg),
        "joint_labels": labels.tolist(),
        "cluster_method": args.method,
        "feature_mode": args.feature_mode,
        "max_lag": args.max_lag,
        "enforce_chain_constraints": args.enforce_chain_constraints,
        "add_contact_part": args.add_contact_part,
        "n_samples": len(motions),
        "n_frames_full": int(activation_mat.shape[0]),
        "n_frames_used": int(activation_sample.shape[0]),
        "chain_fixes": post_changes,
    }

    partition2_path = pjoin(args.output_dir, "skeleton_partition2.json")
    with open(partition2_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if args.sync_primary_partition:
        partition1_path = pjoin(args.output_dir, "skeleton_partition.json")
        with open(partition1_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[info] Synced primary partition: {partition1_path}")

    np.savez(
        pjoin(args.output_dir, "skeleton_stats.npz"),
        mean=mean,
        std=std,
        var=var,
        similarity=similarity,
        activation_sample=activation_sample,
    )

    save_dendrogram(Z, pjoin(args.output_dir, "dendrogram.png"))
    save_leg_activation_plot(
        motions=motions,
        joint_to_dims=joint_to_dims,
        parent_map=parent_map,
        feature_mode=args.feature_mode,
        out_path=pjoin(args.output_dir, "leg_chain_activation_example.png"),
    )

    build_report(
        args=args,
        motions=motions,
        labels=labels,
        part_seg=part_seg,
        part_joint_list=part_joint_list,
        post_changes=post_changes,
        similarity=similarity,
        output_dir=args.output_dir,
    )

    print("[7/7] Done.")
    print(f"Partition saved: {partition2_path}")
    print(f"Report saved: {pjoin(args.output_dir, 'partition_report.txt')}")


if __name__ == "__main__":
    main()



'''
conda run -n tlcontrol python /scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/analyze_skeleton_partition.py \
  --data_root /scratch/ts1v23/workspace/part-aware-vqvae/dataset/HumanML3D \
  --train_split train.txt \
  --n_parts 6 \
  --feature_mode relative_parent \
  --max_lag 4 \
  --add_contact_part \
  --sync_primary_partition \
  --output_dir /scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis

'''