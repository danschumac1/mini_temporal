#!/bin/bash

# Define the base path for models
base_path="/home/dan/mini_temporal/training/models"

# Define the datasets and models
datasets=("AQA") # Add "AQA" if needed
base_models=("gemma-7b" "gemma-2b") # Add other models if needed
trained_contexts=("mixed_context" "relevant_context" "no_context" "random_context" "wrong_date_context") # "relevant_context" "no_context" "random_context" "wrong_date_context"
eval_contexts=("relevant_context" "no_context" "random_context" "wrong_date_context")
# Log file to capture the script's progress
log_file="./generation_log1.txt"
echo "Generation script started at $(date)" > ${log_file}

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Loop through each base model
  for base_model in "${base_models[@]}"; do
    # Loop through each context the model was trained on
    for trained_context in "${trained_contexts[@]}"; do
      # Loop through each context for evaluation
      for eval_context in "${eval_contexts[@]}"; do
        # Define the output directory and file
        output_dir="./output/${dataset}/${base_model//./_}/${trained_context}_trained/${eval_context}_evaluated"
        output_file="${dataset}_${base_model//./_}_${trained_context}_trained_${eval_context}_evaluated.jsonl"
        
        # Create the output directory if it doesn't exist
        mkdir -p "${output_dir}"

        # Log the current combination
        echo "Processing ${dataset}, model ${base_model}, trained on ${trained_context}, evaluated with ${eval_context}" >> ${log_file}

        # Execute the generation script without nohup in the background
        CUDA_VISIBLE_DEVICES=1 python /home/dan/mini_temporal/generations/generations_nit.py \
          --dataset ${dataset} \
          --context ${eval_context} \
          --batch_size 4 \
          --model_path ${base_path}/${dataset}/${base_model//./_}/${trained_context} > "${output_dir}/${output_file}" 2>&1

        # Log the completion of the current combination
        echo "Completed ${dataset}, model ${base_model}, trained on ${trained_context}, evaluated with ${eval_context}" >> ${log_file}
      done
    done
  done
done

echo "Generation script ended at $(date)" >> ${log_file}

#### FOR DEBUGGING -------------------------------------------------
# nohup sh -c "CUDA_VISIBLE_DEVICES=1 python generations_nit.py \
# --dataset TQE \
# --context no_context \
# --batch_size 32 \
# --model_path /home/dan/mini_temporal/training/models/TQE/gemma-7b/no_context" > "trial.jsonl" 2>&1 
### ----------------------------------------------------------------