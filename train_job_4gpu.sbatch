#!/bin/bash
#SBATCH --job-name=trn4
#SBATCH --partition=1day
#SBATCH --gres=gpu:4             # 4 GPUs: train.py splits every batch between them (nn.DataParallel)
#SBATCH --cpus-per-task=32       # 8 CPU cores per GPU, otherwise data loading becomes the bottleneck
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load required modules
ml PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
ml torchvision/0.22.1-foss-2024a-CUDA-12.6.0
ml tqdm/4.66.5-GCCcore-13.3.0   # must be built with the same GCCcore (13.3.0) as PyTorch, otherwise Python is swapped to 3.13 and torch fails to import

# Run the training script from the directory you submitted from.
# `batch_size` and `num_workers` in the config are per GPU: the real values are 4x larger (so num_workers <= 8, as
# 32 CPU cores / 4 GPUs)
# To log to Weights & Biases (optional, see README), change the line below to: python train.py --wandb
python train.py
