#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna or vicuna_guanaco
export string=$2 # number or string
export method=$3 # random or ll
export str_length=$4 # str length: 3, 4, 5
export data_offset=$5 # to spawn several jobs: 0 10 20 30 40 50 60 70 80 90
export seed=$6
export n_train_data=$7
export n_steps=$8


DIR_LOG="/mnt/hdd-nfs/mgubri/adv-suffixes/detect_llm/logs/method_${method}/type_${string}/str_length_${str_length}/model_${model}"
mkdir -p "${DIR_LOG}"

python -u main.py \
    --config="configs/individual_${model}.py" \
    --config.attack=gcg \
    --config.train_data="data/method_${method}/type_${string}/str_length_${str_length}/prompt_goal_n100_seed${seed}.csv" \
    --config.result_prefix="/mnt/hdd-nfs/mgubri/adv-suffixes/detect_llm/results/method_${method}/type_${string}/str_length_${str_length}/model_${model}/gcg_seed${seed}_offset${data_offset}" \
    --config.n_train_data=$n_train_data \
    --config.data_offset=$data_offset \
    --config.n_steps=$n_steps \
    --config.test_steps=10 \
    --config.batch_size=512 \
    --config.stop_on_success=False \
    --config.return_best_loss=True \
    --config.filter_tokens_csv="data/filter_tokens/filter_token_${string}_${model}.csv" >> "${DIR_LOG}/gcg_offset${data_offset}_$(date '+%Y-%m-%d-%H%M%S').log" 2>&1


echo 'DONE!'
# keep best iter, do not stop on success
