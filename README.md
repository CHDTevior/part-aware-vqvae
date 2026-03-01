# Part-Aware VQVAE

This repo packages your VAE/VQ training + evaluation code from `MMM` together with data-driven skeleton partition analysis from `MaskControl`.

## Included
- VQ/VAE training: `train_vq.py`
- VQ smoke/eval script: `test_vq.py`
- Slurm launcher: `train_vq_sbatch.sh`
- Core modules: `models/`, `dataset/`, `utils/`, `options/`, `exit/`
- Partition analysis:
  - script: `partition_analysis/analyze_skeleton_partition.py`
  - partitions: `partition_analysis/skeleton_partition.json`, `partition_analysis/skeleton_partition2.json`

## Quick Start
1. Create environment from `environment.yml`.
2. For large datasets, use symlinks instead of copying:

```bash
scripts/link_large_data.sh /path/to/HumanML3D
```

3. Put checkpoints in paths expected by the original code (same as MMM).
4. Run partition analysis if needed:

```bash
python partition_analysis/analyze_skeleton_partition.py \
  --data_root ./dataset/HumanML3D \
  --output_dir ./partition_analysis
```

5. Train part-aware VQ-VAE:

```bash
python -u train_vq.py \
  --dataname t2m \
  --exp-name vq_data_driven \
  --partition-file ./partition_analysis/skeleton_partition.json \
  --stride-t 2
```

6. Slurm run:

```bash
sbatch train_vq_sbatch.sh
```

## Notes
- Code is intentionally kept close to your original implementation for reproducibility.
- `partition-file` is optional. Without it, the fallback static partition in `models/vqvae.py` is used.
