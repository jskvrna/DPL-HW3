#!/bin/bash
#SBATCH --job-name=trn
#SBATCH --partition=1day
#SBATCH --gres=gpu:1             # number of GPUs; with more, train.py splits every batch between them (also raise --cpus-per-task: 8 per GPU)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load required modules
ml PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
ml torchvision/0.22.1-foss-2024a-CUDA-12.6.0
ml tqdm/4.66.5-GCCcore-13.3.0   # must be built with the same GCCcore (13.3.0) as PyTorch, otherwise Python is swapped to 3.13 and torch fails to import

# Run the training script from the directory you submitted from (`num_workers` in the config is per GPU: <= --cpus-per-task / number of GPUs)
# To log to Weights & Biases (optional, see README), change the line below to: python train.py --wandb
python train.py
