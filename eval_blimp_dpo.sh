#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_blimp
#SBATCH --output=eval_blimp_%A_%a.out
#SBATCH --array=0

MODEL_PATH=/scratch/eo41/babylm/dpo/checkpoint-2400

# execute eval
srun python -u lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 16 \
    --output_path results/blimp/dpo/blimp_results.json