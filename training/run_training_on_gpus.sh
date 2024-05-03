#!/bin/bash





# =============================================================================
# TRAININGS
# =============================================================================

# # NO
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
# --train_data_file ./data/final/train/train_no_context_packed.jsonl \
# --eval_data_file ./data/final/dev/packed/dev_no_context_packed.jsonl \
# --model_context no \
# --epochs 6 \
# --batch_size 4' > run_training_log.txt 2>&1 &


# # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
# --train_data_file ./data/preprocessed/train/train_GEMMA_relevant_context_packed.jsonl \
# --eval_data_file ./data/preprocessed/dev/packed/dev_GEMMA_relevant_context_packed.jsonl \
# --model_context rel \
# --epochs 6 \
# --batch_size 1' > run_training_log.txt 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
# --train_data_file ./data/preprocessed/train/train_GEMMA_random_context_packed.jsonl \
# --eval_data_file ./data/preprocessed/dev/packed/dev_GEMMA_random_context_packed.jsonl \
# --model_context random \
# --epochs 6 \
# --batch_size 1' > run_training_log.txt 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
# --train_data_file ./data/preprocessed/train/train_GEMMA_wrong_date_context_packed.jsonl \
# --eval_data_file ./data/preprocessed/dev/packed/dev_GEMMA_wrong_date_context_packed.jsonl \
# --model_context wd \
# --epochs 6 \
# --batch_size 1' > run_training_log.txt 2>&1 &