CUDA_VISIBLE_DEVICES=0 python fine_tune_gemma_it.py --dataset AQA --epochs 1 --model_context "relevant_context" --base_model gemma-1.1-2b-it --batch_size 16
