#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_blimp
#SBATCH --output=eval_blimp_%A_%a.out
#SBATCH --array=0-25

# define models
MODEL_NAMES=(
    "babylm_10M"
    "wikipedia_10M_1" "wikipedia_10M_2" "wikipedia_10M_3"
    "gutenberg_10M_1" "gutenberg_10M_2" "gutenberg_10M_3"
    "tinystories_10M_1" "tinystories_10M_2" "tinystories_10M_3"
    "pythonedu_10M_1" "pythonedu_10M_2" "pythonedu_10M_3"
    "babylm_100M"
    "wikipedia_100M_1" "wikipedia_100M_2" "wikipedia_100M_3"
    "gutenberg_100M_1" "gutenberg_100M_2" "gutenberg_100M_3"
    "tinystories_100M_1" "tinystories_100M_2" "tinystories_100M_3"
    "pythonedu_100M_1" "pythonedu_100M_2" "pythonedu_100M_3"
)

# get current model name based on array index
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
BASE_MODEL_PATH=/scratch/eo41/babylm/models/${MODEL_NAME}

# loop through step directories within current model
for FULL_MODEL_PATH in ${BASE_MODEL_PATH}/step_*; do

    echo "Model: ${FULL_MODEL_PATH}"

    # extract step name (e.g., step_500)
    STEP_NAME=$(basename "$FULL_MODEL_PATH")

    # execute eval
    srun python -u lm_eval --model hf \
        --model_args pretrained=$FULL_MODEL_PATH,dtype=bfloat16 \
        --tasks blimp_filtered,blimp_supplement \
        --device cuda:0 \
        --batch_size 16 \
        --output_path results/blimp/${MODEL_NAME}/${STEP_NAME}/blimp_results.json
done