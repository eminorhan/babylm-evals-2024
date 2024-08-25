#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_blimp
#SBATCH --output=eval_blimp_%A_%a.out
#SBATCH --array=1-12

# Define the base path for models
BASE_MODEL=babylm_10M_llama
BASE_PATH=/vast/eo41/babylm/models/${BASE_MODEL}

# Compute the step number based on the array index
STEP=$((100 * SLURM_ARRAY_TASK_ID))

# Construct the model path
MODEL_PATH=${BASE_PATH}/step_${STEP}
MODEL_BASENAME=$(basename $MODEL_PATH)

# Execute the evaluation script
srun python -u lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 16 \
    --output_path results/blimp/${BASE_MODEL}/${MODEL_BASENAME}/blimp_results.json

# use `--model_args pretrained=$MODEL_PATH,backend="mlm"` if you're using a custom masked LM
# add --trust_remote_code if you need to load custom config/model files
