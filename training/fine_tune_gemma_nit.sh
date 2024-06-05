#!/bin/bash
# tqe_it_log="TQE_G7B_1E_FATEMEH.out"

#  CUDA_VISIBLE_DEVICES=1 nohup python fine_tune_gemma_nit.py --epochs 1 --dataset TQE --model_context no_context --base_model gemma-7b --batch_size 16 >> $tqe_it_log 2>&1


# Array of context types
#contexts=("no_context") #  "wrong_date_context"  "random_context" "relevant_context"
contexts=("wrong_date_context" "random_context" "relevant_context")

# Array of base models
base_models=("gemma-2b" "gemma-7b")

# Log files
# tqe_it_log="TQE_G7B_1E_FATEMEH.out"
aqa_nit_log="AQA-NIT_LOGv2.out"

# Clear the log file at the start
# echo "" > $tqe_it_log

# # TQE HAS THE DEFAULT EPOCHS
# for base_model in "${base_models[@]}"; do
#   for context in "${contexts[@]}"; do
#     echo "\n\nTQE: TRAINING $base_model WITH $context\n\n" | tee -a $tqe_it_log
#     nohup bash -c "CUDA_VISIBLE_DEVICES=1 python fine_tune_gemma_nit.py --epochs 1 --dataset TQE --model_context $context --base_model $base_model --batch_size 16" >> $tqe_it_log 2>&1
#   done
# done

echo "" > $aqa_nit_log

for base_model in "${base_models[@]}"; do
  for context in "${contexts[@]}"; do
    echo "\n\nAQA: TRAINING $base_model WITH $context\n\n"
    nohup bash -c "CUDA_VISIBLE_DEVICES=0 python fine_tune_gemma_nit.py --epochs 1 --dataset AQA --model_context $context --base_model $base_model --batch_size 16 >> $aqa_nit_log 2>&1" 
  done
done
