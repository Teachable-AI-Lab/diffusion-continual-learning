#!/bin/bash
#SBATCH --account=gts-cmaclellan3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=L40s:1
#SBATCH --mem-per-cpu=16G
#SBATCH -qinferno
#SBATCH -t16:00:00

module load python cuda
source ~/p-cmaclellan3-0/iclr-env/bin/activate
cd ~/p-cmaclellan3-0/diffusion-continual-learning

python train-model.py --config=test-config-gr-dist.json