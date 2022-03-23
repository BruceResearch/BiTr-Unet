#!/bin/bash
#
#SBATCH --job-name=JQR
#SBATCH --nodes=1
#SBATCH --mem=28GB
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

module load python/intel/3.8.6
module load cuda/11.1.74

python3.8 evaluation.py