#!/bin/bash
#SBATCH -p swarm_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=vq_data_driven
#SBATCH --output=logs/train_vq_data_driven_%j.log
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G

# 激活 Conda 环境
source ~/.bashrc
conda activate tlcontrol

# 设置工作目录
cd /scratch/ts1v23/workspace/part-aware-vqvae
mkdir -p logs

echo "Job started at $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS_ON_NODE"

# 启动 VQ-VAE 训练（数据驱动 skeleton 划分）
python -u train_vq.py \
    --dataname t2m \
    --exp-name vq_data_driven \
    --partition-file ./partition_analysis/skeleton_partition.json \
    --stride-t 2 \
    "$@"

echo "Job finished at $(date)"
