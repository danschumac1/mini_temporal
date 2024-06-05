#!/bin/bash

# Array of base models
nit_models=("gemma-2b" "gemma-7b")
it_models=("gemma-1.1-2b-it" "gemma-1.1-7b-it")

# Set up log files
mixed_nit_log="mixed_nit_log.out"
mixed_it_log="mixed_it_log.out"

# Clear outputs and make sure files exists
echo "" > $mixed_nit_log
echo "" > $mixed_it_log

for nit_model in "${nit_models[@]}"; do
  echo -e "\n\nAQA: TRAINING $nit_model WITH mixed context!\n\n" | tee -a $mixed_nit_log
  CUDA_VISIBLE_DEVICES=0 nohup python fine_tune_gemma_MIXED_nit.py \
      --epochs 6 \
      --dataset TQE \
      --model_context mixed_context \
      --base_model $nit_model \
      --batch_size 16 >> $mixed_nit_log 2>&1
done

# WAIT WAIT WAIT

for it_model in "${it_models[@]}"; do
  echo -e "\n\nAQA: TRAINING $it_model WITH mixed context!\n\n" | tee -a $mixed_it_log
  CUDA_VISIBLE_DEVICES=0 nohup python fine_tune_gemma_MIXED_it.py \
      --epochs 6 \
      --dataset TQE \
      --model_context mixed_context \
      --base_model $it_model \
      --batch_size 16 >> $mixed_it_log 2>&1
done
