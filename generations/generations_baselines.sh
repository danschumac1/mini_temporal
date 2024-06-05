#!/bin/bash

# =============================================================================
# SET UP
# =============================================================================
# Define the base path for models
base_path="/home/dan/mini_temporal/training/models"

# Define the datasets and models
datasets=("TQE") # Add "AQA" if needed
NIT_models=("gemma-7b" "gemma-2b")
IT_models=("gemma-1.1-7b-it" "gemma-1.1-2b-it")
contexts=("relevant_context" "no_context" "random_context" "wrong_date_context")

# =============================================================================
# IT
# =============================================================================
IT_file="./baseline_it_generations_logging.txt"
echo "Generation script started at $(date)" > ${IT_file}

for dataset in "${datasets[@]}"; do
  # Loop through each base model
  for IT_model in "${IT_models[@]}"; do   
    # Loop through each context for evaluation
    for eval_context in "${contexts[@]}"; do
      # Define the output directory and file
      output_dir="./output/${dataset}/baseline/${IT_model}/${eval_context}"
      output_file="${dataset}_baseline_${IT_model}_${eval_context}_evaluated.jsonl"
      
      # Create the output directory if it doesn't exist
      mkdir -p "${output_dir}"

      # Log the current combination
      echo "Processing ${dataset}, model ${IT_model}, evaluated with ${eval_context}" >> ${IT_file}

      # Execute the generation script without nohup in the background
      CUDA_VISIBLE_DEVICES=0 python /home/dan/mini_temporal/generations/generations_baselines.py \
        --dataset ${dataset} \
        --context ${eval_context} \
        --batch_size 4 \
        --model_path ${IT_model} > "${output_dir}/${output_file}" 2>&1

      # Log the completion of the current combination
      echo "Completed ${dataset}, model ${IT_model}, evaluated with ${eval_context}" >> ${IT_file}
    done
  done
done

echo "IT script ended at $(date)" >> ${IT_file}

# =============================================================================
# NIT
# =============================================================================
NIT_file="./baseline_nit_generations_logging.txt"
echo "Generation script started at $(date)" > ${NIT_file}

for dataset in "${datasets[@]}"; do
  # Loop through each base model
  for NIT_model in "${NIT_models[@]}"; do   
    # Loop through each context for evaluation
    for eval_context in "${contexts[@]}"; do
      # Define the output directory and file
      output_dir="./output/${dataset}/baseline/${NIT_model}/${eval_context}"
      output_file="${dataset}_baseline_${NIT_model}_${eval_context}_evaluated.jsonl"
      
      # Create the output directory if it doesn't exist
      mkdir -p "${output_dir}"

      # Log the current combination
      echo "Processing ${dataset}, model ${NIT_model}, evaluated with ${eval_context}" >> ${NIT_file}

      # Execute the generation script without nohup in the background
      CUDA_VISIBLE_DEVICES=0 python /home/dan/mini_temporal/generations/generations_baselines.py \
        --dataset ${dataset} \
        --context ${eval_context} \
        --batch_size 4 \
        --model_path ${NIT_model} > "${output_dir}/${output_file}" 2>&1

      # Log the completion of the current combination
      echo "Completed ${dataset}, model ${NIT_model}, evaluated with ${eval_context}" >> ${NIT_file}
    done
  done
done

echo "NIT script ended at $(date)" >> ${NIT_file}

#### FOR DEBUGGING -------------------------------------------------
# nohup sh -c "CUDA_VISIBLE_DEVICES=1 python generations_nit.py \
# --dataset TQE \
# --context no_context \
# --batch_size 32 \
# --model_path /home/dan/mini_temporal/training/models/TQE/gemma-7b/no_context" > "trial.jsonl" 2>&1 
### ----------------------------------------------------------------
