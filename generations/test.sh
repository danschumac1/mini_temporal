CUDA_VISIBLE_DEVICES=0 python generations_nit.py \
--dataset TQE \
--context relevant_context \
--batch_size 1 \
--model_path /home/dan/mini_temporal/training/models/TQE/gemma-7b/relevant_context > "trial-it2.jsonl"
#--model_path /home/dan/mini_temporal/training/models/TQE/gemma-1_1-7b-it/relevant_context > "trial2.jsonl"
