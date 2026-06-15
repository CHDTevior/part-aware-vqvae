# Part-Aware VQ-VAE

This repository contains the part-aware VQ-VAE tokenizer code used for the
overlap-partition HumanML3D/T2M run `vq_overlap_top3_20260529`.

This is the VQ-VAE repository only. The continuous PartVAE work is kept in the
separate repository `CHDTevior/part-aware-partvae`.

## Current Reference Checkpoint

Reference run:

```text
vq_overlap_top3_20260529
```

Training output directory in the original workspace:

```text
./output/vq/2026-05-29-11-07-24_vq_overlap_top3_20260529
```

Public Hugging Face bundle with weights, configs, logs, and code snapshot:

```text
https://huggingface.co/Tevior/part-aware-vqvae-overlap-top3-20260529
```

GitHub release assets mirror the checkpoint files:

```text
https://github.com/CHDTevior/part-aware-vqvae/releases/tag/vq-overlap-top3-20260529
```

Use `net_best_fid.pth` by default for reconstruction-FID comparisons. Use
`net_best_top3.pth` only when selecting for R@3.

## Checkpoints

The released bundle contains:

```text
net_best_fid.pth
net_best_div.pth
net_best_top1.pth
net_best_top3.pth
net_best_matching.pth
net_last.pth
```

The HF repo also includes:

```text
config/training_config.json
config/training_eval_summary.json
config/skeleton_partition.json
config/launch_command.txt
logs/run.log
logs/train_vq_overlap_top3_20260529_20260529_100714.log
manifest.sha256
```

For convenience, the config and partition are duplicated next to the HF
`weights/` directory as `weights/config.json` and
`weights/skeleton_partition.json`.

## Metrics

These are training-time reconstruction metrics parsed from `run.log`. They are
not a formal 5-seed text-generation benchmark.

| checkpoint / selector | iter | FID | R@1 | R@2 | R@3 | Matching | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `net_best_fid.pth` | 165000 | 0.0153 | 0.5153 | 0.7021 | 0.7965 | 2.9400 | 9.6057 |
| `net_best_top1.pth` | 210000 | 0.0219 | 0.5372 | 0.7081 | 0.8012 | 2.9300 | 9.6410 |
| `net_best_top3.pth` | 60000 | 0.0366 | 0.5160 | 0.7154 | 0.8112 | 2.9063 | 9.3176 |
| `net_best_matching.pth` | 160000 | 0.0426 | 0.5312 | 0.7114 | 0.8052 | 2.8848 | 9.5794 |
| `net_last.pth` | 300000 | 0.0209 | 0.5140 | 0.7114 | 0.7999 | 2.9161 | 9.3756 |

Final train line:

```text
iter=300000, Recons=0.02611, PPL=83.99, Commit=0.89492
```

## Exact Training Config

```text
dataname=t2m
exp_name=vq_overlap_top3_20260529
partition_file=partition_analysis/skeleton_partition.json
nb_code=128
code_dim=128
output_emb_width=128
batch_size=256
window_size=64
lr=2e-4
total_iter=300000
warm_up_iter=1000
lr_scheduler=[200000]
gamma=0.05
weight_decay=0.0
commit=0.02
loss_vel=0.5
recons_loss=l1_smooth
down_t=2
stride_t=2
width=512
depth=3
dilation_growth_rate=3
vq_act=relu
vq_norm=null
quantizer=ema_reset
mu=0.99
beta=1.0
sep_uplow=false
seed=123
eval_iter=5000
print_iter=200
```

Important: this checkpoint uses the overlap JSON partition
`partition_analysis/skeleton_partition.json`. Do not run it as the MMM
hardcoded/default-partition checkpoint, and do not leave `--partition-file`
empty for this run.

## Environment

The original environment was `tlcontrol`. A reproducibility snapshot is stored
in `environment.yml`.

Minimal setup:

```bash
conda env create -f environment.yml
conda activate tlcontrol
```

If you build a lighter environment manually, you still need PyTorch, NumPy,
SciPy, scikit-learn, tqdm, TensorBoard, and the HumanML3D evaluator
dependencies used by this codebase.

## Data and Evaluator Assets

The training and eval scripts expect the same local layout as the original
project:

```text
dataset/HumanML3D/
glove/
checkpoints/t2m/Comp_v6_KLD005/
checkpoints/t2m/text_mot_match/model/finest.tar
```

`dataset/HumanML3D/` should contain the HumanML3D motion/text split files used
by the original MMM/T2M setup. `glove/` should contain
`our_vab_*.{npy,pkl}`. The evaluator checkpoints are required for FID,
diversity, R-precision, and matching score.

## Reproduce Training

From the repository root:

```bash
bash scripts/launch_vq_overlap_top3_20260529.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_vq.py \
  --dataname t2m \
  --seed 123 \
  --exp-name vq_overlap_top3_20260529 \
  --nb-code 128 \
  --partition-file ./partition_analysis/skeleton_partition.json
```

The historical launch used the absolute partition path from the original
workspace:

```text
./partition_analysis/skeleton_partition.json
```

The JSON contents are included in this repository and in the HF bundle.

## Offline Evaluation

Download the HF bundle or place a checkpoint under `output/vq/...`, then run:

```bash
python scripts/offline_vq_eval.py \
  --ckpt /path/to/net_best_fid.pth \
  --partition-file ./partition_analysis/skeleton_partition.json \
  --nb-code 128 \
  --code-dim 128 \
  --output-emb-width 128 \
  --down-t 2 \
  --stride-t 2 \
  --run-training-style-eval \
  --out-dir offline_eval/vq_overlap_top3_20260529_best_fid
```

For latent/codebook diagnostics without the full training-style eval:

```bash
python scripts/offline_vq_eval.py \
  --ckpt /path/to/net_best_fid.pth \
  --partition-file ./partition_analysis/skeleton_partition.json \
  --nb-code 128 \
  --code-dim 128 \
  --output-emb-width 128 \
  --down-t 2 \
  --stride-t 2 \
  --num-batches 40 \
  --out-dir offline_eval/vq_overlap_top3_20260529_latent_diag
```

## Loading Checkpoint

```python
import json
import torch
from types import SimpleNamespace

from models.vqvae import HumanVQVAE

with open("configs/vq_overlap_top3_20260529_training_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

args = SimpleNamespace(**cfg)
args.partition_file = "./partition_analysis/skeleton_partition.json"

model = HumanVQVAE(
    args,
    nb_code=cfg["nb_code"],
    code_dim=cfg["code_dim"],
    output_emb_width=cfg["output_emb_width"],
    down_t=cfg["down_t"],
    stride_t=cfg["stride_t"],
    width=cfg["width"],
    depth=cfg["depth"],
    dilation_growth_rate=cfg["dilation_growth_rate"],
    activation=cfg["vq_act"],
    norm=cfg["vq_norm"],
)

ckpt = torch.load("weights/net_best_fid.pth", map_location="cpu", weights_only=False)
state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.eval()
```

The current quantizer implementation allocates its codebook with `.cuda()` at
initialization time, matching the original training code. Instantiate on a CUDA
machine for normal use.

## Repository Files Added for This Release

```text
configs/vq_overlap_top3_20260529_training_config.json
configs/vq_overlap_top3_20260529_training_eval_summary.json
configs/vq_overlap_top3_20260529_launch_command.txt
configs/vq_overlap_top3_20260529_skeleton_partition.json
docs/vq_overlap_top3_20260529_metrics.md
docs/vq_overlap_top3_20260529_hf_model_card.md
scripts/launch_vq_overlap_top3_20260529.sh
```

`utils/eval_trans.py` is updated so the R@3 selector also saves
`net_best_top3.pth`.
