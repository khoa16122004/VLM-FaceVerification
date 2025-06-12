#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
 python "[script]eval_bench.py" --img_dir lfw/images --pair_path lfw/1000_pair.txt --pretrained_lvlm llava-onevision-qwen2-7b-ov --model_name_lvlm llava_qwen --llm_model Llama-7b --num_samples 9