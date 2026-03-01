# Skeleton Partition2 Analysis Report

**Data source:** `skeleton_partition2.json`  
**Generated from:** HumanML3D data-driven skeleton partitioning

---

## 1. Metadata Summary

| Field | Value | Description |
|-------|-------|-------------|
| n_parts | 6 | Number of body part segments |
| n_samples | 23,368 | Motion samples used for correlation computation (full train split) |
| cluster_method | average | Hierarchical clustering method (average linkage) |

---

## 2. Part Dimension Distribution

Based on `partSeg` in the JSON:

| Part | Dimensions | Share | Body Region |
|------|------------|-------|-------------|
| Part 0 | 143 | 54% | Body core |
| Part 1 | 24 | 9% | Left wrist / left hand |
| Part 2 | 24 | 9% | Right wrist / right hand |
| Part 3 | 24 | 9% | Left ankle / left toe |
| Part 4 | 24 | 9% | Right ankle / right toe |
| Part 5 | 24 | 9% | Left knee / right knee |
| **Total** | **263** | 100% | |

---

## 3. Joint-to-Cluster Mapping (joint_labels)

| Joint ID | Anatomical Body Part | Cluster ID |
|----------|----------------------|------------|
| 0 | root (pelvis) | 1 |
| 1, 2 | left hip, right hip | 1 |
| 3, 6, 9, 12 | spine (lower, mid, upper, neck) | 1 |
| 13, 14 | left shoulder, right shoulder | 1 |
| 15 | head | 1 |
| 16, 17 | left elbow, right elbow | 1 |
| 4, 5 | left knee, right knee | 6 |
| 7, 10 | left ankle, left toe | 4 |
| 8, 11 | right ankle, right toe | 5 |
| 18, 20 | left wrist, left hand | 2 |
| 19, 21 | right wrist, right hand | 3 |

---

## 4. Part-to-Body-Region Correspondence

| Part | Dims | Joint IDs | Body Region |
|------|-----|-----------|-------------|
| Part 0 | 143 | 0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17 | Root, hips, spine, neck, head, shoulders, elbows + foot contact (259–262) |
| Part 1 | 24 | 18, 20 | Left wrist, left hand |
| Part 2 | 24 | 19, 21 | Right wrist, right hand |
| Part 3 | 24 | 7, 10 | Left ankle, left toe |
| Part 4 | 24 | 8, 11 | Right ankle, right toe |
| Part 5 | 24 | 4, 5 | Left knee, right knee |

---

## 5. Comparison with Manual VAE Partitioning

Default `partSeg` in `models/vqvae.py` (lines 254–261):

| Manual Partition | Data-Driven Partition |
|------------------|------------------------|
| Root (separate) | Merged into Part 0 (body core) |
| Spine (3, 6, 9, 12, 15) | Part 0 |
| Left arm (13, 16, 18, 20) | Part 0 (shoulder, elbow) + Part 1 (wrist, hand) |
| Right arm (14, 17, 19, 21) | Part 0 (shoulder, elbow) + Part 2 (wrist, hand) |
| Left leg (1, 4, 7, 10) + left foot | Part 0 (hip) + Part 5 (knee) + Part 3 (ankle, toe) + left foot contact |
| Right leg (2, 5, 8, 11) + right foot | Part 0 (hip) + Part 5 (knee) + Part 4 (ankle, toe) + right foot contact |

**Manual partition:** Anatomical chains (root / spine / left arm / right arm / left leg / right leg), roughly balanced in size.  
**Data-driven partition:** Clustering by motion correlation; core (root, spine, hips, shoulders, elbows) merged into one large part; limbs split into distal segments (wrists/hands, ankles/toes).

---

## 6. Physical Interpretation and Conclusions

- **Body core (Part 0):** Root, spine, hips, shoulders, and elbows show strong temporal correlation and are grouped into one cluster.
- **Both knees (Part 5):** Left and right knees exhibit symmetric motion in walking, squatting, etc., forming a separate cluster.
- **Distal limbs (Parts 1–4):** Wrists/hands and ankles/toes have different motion patterns from proximal joints and are separated into distinct clusters.
- **Foot contact (259–262):** Assigned to Part 0 (which contains hip joints).

The data-driven partition is finer than the manual one: knees, left/right lower legs, and left/right hands are isolated, while hips, shoulders, and elbows are merged with the torso, consistent with motion correlation statistics.
