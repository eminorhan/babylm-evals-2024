#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_glue
#SBATCH --output=eval_glue_%A_%a.out
#SBATCH --array=0

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

	STEP_NAME=$(basename $FULL_MODEL_PATH)
	for TASK in {boolq,cola,mnli,mnli-mm,mrpc,multirc,qnli,qqp,rte,sst2,wsc}; do
		if [[ $TASK = "mnli-mm" ]]; then
			TRAIN_NAME="mnli"
			VALID_NAME="mnli-mm"
			DO_TRAIN=False
			MODEL_PATH_FULL="results/finetune/$MODEL_NAME/$STEP_NAME/$TRAIN_NAME/"
		else
			TRAIN_NAME=$TASK
			VALID_NAME=$TASK
			DO_TRAIN=True
			MODEL_PATH_FULL=$FULL_MODEL_PATH
		fi

		mkdir -p results/finetune/$MODEL_NAME/$STEP_NAME/$TASK/

		python finetune_classification.py \
		--model_name_or_path $MODEL_PATH_FULL \
		--output_dir results/finetune/$MODEL_NAME/$STEP_NAME/$TASK/ \
		--train_file evaluation_data/glue_filtered/$TRAIN_NAME.train.jsonl \
		--validation_file evaluation_data/glue_filtered/$VALID_NAME.valid.jsonl \
		--do_train $DO_TRAIN \
		--do_eval \
		--do_predict \
		--max_seq_length 128 \
		--per_device_train_batch_size 64 \
		--learning_rate 0.0003 \
		--num_train_epochs 10 \
		--patience 3 \
		--evaluation_strategy epoch \
		--save_strategy epoch \
		--overwrite_output_dir \
		--trust_remote_code \
		--seed 1
	done
done