# Training Metrics

These are training-time VQ-VAE evaluation metrics parsed from `logs/run.log`. No formal offline 5-seed evaluation is bundled here.

| checkpoint / selector | iter | FID | R@1 / R@2 / R@3 | Matching | Diversity |
|---|---:|---:|---:|---:|---:|
| `net_best_fid.pth` / best_fid | 165000 | 0.0153 | 0.5153 / 0.7021 / 0.7965 | 2.9400 | 9.6057 |
| `net_best_top1.pth` / best_top1 | 210000 | 0.0219 | 0.5372 / 0.7081 / 0.8012 | 2.9300 | 9.6410 |
| `net_best_top3.pth` / best_top3 | 60000 | 0.0366 | 0.5160 / 0.7154 / 0.8112 | 2.9063 | 9.3176 |
| `net_best_matching.pth` / best_matching | 160000 | 0.0426 | 0.5312 / 0.7114 / 0.8052 | 2.8848 | 9.5794 |
| `net_last.pth` / last | 300000 | 0.0209 | 0.5140 / 0.7114 / 0.7999 | 2.9161 | 9.3756 |

Final train line: iter 300000, Recons 0.02611, PPL 83.99, Commit 0.89492.
