#!/bin/bash
echo $1
sbatch --job-name=$1 --output=logs/$1.out --error=logs/$1.err job.sh $1

