#!/bin/bash

# Array of context types
contexts=("wrong_date_context" "no_context" "random_context" "relevant_context") #  

# Array of base models
base_models=("gemma-2b" "gemma-7b") # Add "gemma-1.1-2b-it" if needed

# Log files
tqe_it_log="TQE-IT_LOG.out"
# aqa_it_log="AQA-IT_LOG.out"

# Clear the log file at the start
echo "" > $tqe_it_log

# TQE HAS THE DEFAULT EPOCHS
for base_model in "${base_models[@]}"; do
  for context in "${contexts[@]}"; do
    echo "\n\nTQE: TRAINING $base_model WITH $context\n\n" | tee -a $tqe_it_log
    nohup bash -c "CUDA_VISIBLE_DEVICES=1 python fine_tune_gemma_nit.py --dataset TQE --model_context $context --base_model $base_model --batch_size 16" >> $tqe_it_log 2>&1
  done
done


# echo "" > $aqa_it_log

# AQA ONLY HAS 4 EPOCHS
# for base_model in "${base_models[@]}"; do
#   for context in "${contexts[@]}"; do
#     echo "\n\nAQA: TRAINING $base_model WITH $context\n\n"
#     nohup bash -c "CUDA_VISIBLE_DEVICES=0 python fine_tune_gemma_it.py --epochs 4 --dataset AQA --model_context $context --base_model $base_model >> $aqa_it_log 2>&1" 
#   done
# done