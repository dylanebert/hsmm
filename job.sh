#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
echo $1
conda activate act-recog
python controller.py --model autoencoder --config $1
