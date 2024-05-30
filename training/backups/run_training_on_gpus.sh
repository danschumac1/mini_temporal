#!/bin/bash





# =============================================================================
# gemma-2b-it TRAININGS
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
# --train_data_file ./data/final/train/train_rel_context_packed.jsonl \
# --eval_data_file ./data/final/dev/packed/dev_rel_context_packed.jsonl \
# --model_context rel \
# --epochs 6 \
# --batch_size 1' > run_training_log.txt 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
# --train_data_file ./data/final/train/train_random_context_packed.jsonl \
# --eval_data_file ./data/final/dev/packed/dev_random_context_packed.jsonl \
# --model_context random \
# --epochs 6 \
# --batch_size 1' > run_training_log.txt 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
# --train_data_file ./data/final/train/train_wd_context_packed.jsonl \
# --eval_data_file ./data/final/dev/packed/dev_wd_context_packed.jsonl \
# --model_context wd \
# --epochs 6 \
# --batch_size 1' > run_training_log.txt 2>&1 &

# =============================================================================
# gemma-2b TRAININGS (NOT INSTRUCTION TUNED)
# =============================================================================

# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_no_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_no_context_packed.jsonl \
--model_context no \
--epochs 16 \
--batch_size 2 \
--base_model gemma-2b' > c0_run_training_log.txt 2>&1 &


# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_rel_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_rel_context_packed.jsonl \
--model_context rel \
--epochs 6 \
--batch_size 1 \
--base_model gemma-2b' > c1_run_training_log.txt 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_random_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_random_context_packed.jsonl \
--model_context random \
--epochs 6 \
--batch_size 1 \
--base_model gemma-2b' > c0_run_training_log.txt 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_wd_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_wd_context_packed.jsonl \
--model_context wd \
--epochs 6 \
--batch_size 1 \
--base_model gemma-2b' > c1_run_training_log.txt 2>&1 &

# =============================================================================
# gemma-7b 
# =============================================================================

# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_no_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_no_context_packed.jsonl \
--model_context no \
--epochs 16 \
--batch_size 2 \
--base_model gemma-7b' > c0_run_training_log.txt 2>&1 &


# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_rel_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_rel_context_packed.jsonl \
--model_context rel \
--epochs 6 \
--batch_size 1 \
--base_model gemma-7b' > c1_run_training_log.txt 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_random_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_random_context_packed.jsonl \
--model_context random \
--epochs 6 \
--batch_size 1 \
--base_model gemma-7b' > c0_run_training_log.txt 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python run_training_on_gpus.py \
--train_data_file ./data/final/train/train_wd_context_packed.jsonl \
--eval_data_file ./data/final/dev/packed/dev_wd_context_packed.jsonl \
--model_context wd \
--epochs 6 \
--batch_size 1 \
--base_model gemma-7b' > c1_run_training_log.txt 2>&1 &