#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=16G

python main.py --model unsupervised

