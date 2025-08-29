#!/bin/bash
#SBATCH --account=gts-cmaclellan3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=L40s:1
#SBATCH --mem-per-cpu=16G
#SBATCH -qinferno
#SBATCH -t12:00:00

module load python cuda
source ~/p-cmaclellan3-0/iclr-env/bin/activate
cd ~/p-cmaclellan3-0/diffusion-continual-learning

# python train-model.py --config=test-config-rank1-only.json
python train-model.py --config=test-config-rank1-2k.json
# python train-model.py --config=test-config-rank1.json