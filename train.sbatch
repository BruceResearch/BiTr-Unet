#!/bin/bash
#
#SBATCH --job-name=JQR
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=12
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4

module load python/intel/3.8.6
module load cuda/11.1.74

python3.8 -m torch.distributed.launch --nproc_per_node=4 --master_port 6666 train.py 
