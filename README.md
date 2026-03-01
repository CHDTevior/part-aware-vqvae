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

## Environment (pip-first)
Use `conda` only to create an isolated env, then install packages via `pip`.

```bash
# 1) Create and activate env
conda create -n tlcontrol python=3.10 -y
conda activate tlcontrol

# 2) Install PyTorch (CUDA 11.8 example)
pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# 3) Install project dependencies
pip install -r environment.txt
```

If you use a different CUDA version, replace step 2 with the matching PyTorch command from pytorch.org.

## Quick Start
1. For large datasets, use symlinks instead of copying:

```bash
scripts/link_large_data.sh /path/to/HumanML3D
```

2. Link shared training assets (`glove`, `checkpoints`) from your existing workspace:

```bash
ln -sfn /your/path/to/MMM/glove glove
ln -sfn /your/path/to/MMM/checkpoints checkpoints
```

3. Run partition analysis if needed:

```bash
python partition_analysis/analyze_skeleton_partition.py \
  --data_root ./dataset/HumanML3D \
  --output_dir ./partition_analysis
```

4. Optional smoke test (no full training):

```bash
python scripts/smoke_test.py --device cpu
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
- Real training requires `glove/our_vab_*.{npy,pkl}` and `checkpoints/t2m/Comp_v6_KLD005/opt.txt`.
- If `sbatch` is pending with `QOSMaxNodePerUserLimit`, run inside an existing allocation instead:

```bash
srun --jobid=<running_jobid> --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G \
  bash -lc 'source ~/.bashrc && conda activate tlcontrol && cd /path/to/part-aware-vqvae && python -u train_vq.py ...'
```

- Keep `--ntasks=1` for single-process training to avoid duplicate process collisions on output folders.
