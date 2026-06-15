---
license: other
tags:
- motion-generation
- vqvae
- humanml3d
- part-aware-vqvae
- checkpoint
---

# Part-Aware VQ-VAE Overlap Top3 20260529

Public checkpoint bundle for the `part-aware-vqvae` VQ-VAE rerun `vq_overlap_top3_20260529`.

Source workspace at export time: `.`
Training output directory: `./output/vq/2026-05-29-11-07-24_vq_overlap_top3_20260529`
Git commit recorded at export time: `82d4a3f538ae8e1fc623b7c8694ba32111245062`

## Contents

- `weights/`: all saved training checkpoints: best FID, best diversity, best Top1, best Top3, best matching score, and last.
- `config/training_config.json`: parsed JSON config from the training log plus launch command.
- `config/skeleton_partition.json`: overlap body-part partition used by this run.
- `config/training_eval_summary.json`: parsed training-time eval summary.
- `logs/`: `run.log`, launcher stdout/stderr log, and TensorBoard event file.
- `code/`: minimal code snapshot needed for loading/evaluation/reproduction, plus git status/diff.
- `manifest.sha256`: SHA256 checksums for every uploaded file.

## Reproduce Launch

```bash
cd .
conda activate tlcontrol
CUDA_VISIBLE_DEVICES=0 python -u train_vq.py --dataname t2m --seed 123 --exp-name vq_overlap_top3_20260529 --nb-code 128 --partition-file ./partition_analysis/skeleton_partition.json
```

Important: this is the overlap-partition VQ-VAE (`partition_analysis/skeleton_partition.json`), not the MMM hardcoded/default-partition checkpoint.

## Main Training-Time Metrics

See `METRICS.md` for the parsed table. Best FID checkpoint:

- `weights/net_best_fid.pth`
- iter: 165000
- FID: 0.0153
- R@1/R@2/R@3: 0.5153 / 0.7021 / 0.7965
- matching: 2.9400
- diversity: 9.6057

## Caveat

The metrics bundled here are training-time eval metrics from `run.log`; they are not a formal offline 5-seed evaluation.
