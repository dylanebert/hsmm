#!/bin/bash
##SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=128G
#SBATCH -t 24:00:00
echo $1
conda activate act-recog
python input_modules.py
