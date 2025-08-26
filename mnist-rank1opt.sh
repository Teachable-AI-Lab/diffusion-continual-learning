#!/bin/bash
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40
#SBATCH --mem-per-gpu="16GB"
#SBATCH --qos="short"
#SBATCH --exclude="spot,heistotron"

source ~/.bashrc
conda init
conda activate dl
cd /nethome/zwang910/research/continual-learning

python train-model.py --config=test-config-rank1opt.json