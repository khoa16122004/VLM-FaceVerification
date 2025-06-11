#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
python "[script]eval_bench.py" --start_index 1078 --img_dir lfw/lfw/images --pair_path lfw/lfw/pairs.txt