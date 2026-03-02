# HumanML3D 全样本 Partition 分析报告（Overlap 版）

## 1. 本次实验配置（最新）

命令：

```bash
conda run -n tlcontrol python /scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/analyze_skeleton_partition.py \
  --data_root /scratch/ts1v23/workspace/part-aware-vqvae/dataset/HumanML3D \
  --train_split train.txt \
  --n_parts 6 \
  --feature_mode relative_parent \
  --max_lag 4 \
  --overlap_preset boundary_v1 \
  --sync_primary_partition \
  --output_dir /scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis
```

结果文件：
- [`skeleton_partition.json`](./skeleton_partition.json)
- [`skeleton_partition2.json`](./skeleton_partition2.json)
- [`partition_report.txt`](./partition_report.txt)

关键配置（回读自 JSON）：
- `n_samples = 23368`
- `n_frames_full = 3293312`
- `n_frames_used = 200000`
- `feature_mode = relative_parent`
- `parent_delta_rot = False`
- `parent_delta_vel = False`
- `chain_target_mode = majority`
- `add_contact_part = False`（contact 并入左右腿）
- `overlap_preset = boundary_v1`

## 2. 计算方式

```mermaid
flowchart TD
    A[load motions] --> B[build feat_j(t)]
    B --> C[a_j(t)=||feat_j(t)||_2]
    C --> D[lagged sim max|corr|]
    D --> E[dist=1-sim]
    E --> F[hierarchical clustering]
    F --> G[chain constraints]
    G --> H[cluster balancing]
    H --> I[partSeg no-overlap]
    I --> J[apply overlap preset]
    J --> K[save json/report]
```

核心公式：
- `a_j(t) = ||feat_j(t)||_2`
- `sim(i,j) = max_{tau in [-K,K]} |corr(a_i[t], a_j[t+tau])|`
- `dist(i,j) = 1 - sim(i,j)`，本次 `K=4`

## 3. 最新分区结果（含 overlap）

| Part | no-overlap dims | overlap dims | joints |
|---|---:|---:|---|
| Part 0 | 19 | 19 | root/pelvis, spine1 |
| Part 1 | 108 | 132 | spine2 + 双侧 collar/shoulder/elbow/wrist |
| Part 2 | 50 | 81 | right hip/knee/ankle/foot + contact(261,262) |
| Part 3 | 24 | 60 | spine3, neck |
| Part 4 | 50 | 81 | left hip/knee/ankle/foot + contact(259,260) |
| Part 5 | 12 | 24 | head |

说明：
- JSON 中 `partSeg_no_overlap` 保留了重合前版本。
- JSON 中 `partSeg` 是当前训练将使用的 overlap 版本。

### 3.1 每个 Part 的 Joint 组成（base / overlap / effective）

| Part | base joints | overlap source joints | effective joints |
|---|---|---|---|
| Part 0 | root/pelvis, spine1 | (none) | root/pelvis, spine1 |
| Part 1 | spine2, left/right collar, left/right shoulder, left/right elbow, left/right wrist | spine1, neck | spine1, spine2, neck, left/right collar, left/right shoulder, left/right elbow, left/right wrist |
| Part 2 | right hip, right knee, right ankle, right foot | root/pelvis, spine1, spine2 | root/pelvis, spine1, spine2, right hip, right knee, right ankle, right foot |
| Part 3 | spine3, neck | spine2, left collar, right collar | spine2, spine3, neck, left collar, right collar |
| Part 4 | left hip, left knee, left ankle, left foot | root/pelvis, spine1, spine2 | root/pelvis, spine1, spine2, left hip, left knee, left ankle, left foot |
| Part 5 | head | neck | neck, head |

## 4. Overlap 事件（boundary_v1）

本次共 `12` 条事件，核心桥接为：
- root/spine1/spine2 -> 左右腿上游
- spine2 <-> neck/head
- 左右 collar -> neck

详细事件已写入：
- [`skeleton_partition2.json`](./skeleton_partition2.json) 的 `overlap_events`
- [`partition_report.txt`](./partition_report.txt) 的 `Overlap events` 段落

## 5. 代码位置

- overlap 后处理实现：[`analyze_skeleton_partition.py:95`](./analyze_skeleton_partition.py#L95)
- overlap 参数解析：[`analyze_skeleton_partition.py:687`](./analyze_skeleton_partition.py#L687)
- no-overlap / overlap 双输出：[`analyze_skeleton_partition.py:762`](./analyze_skeleton_partition.py#L762)
