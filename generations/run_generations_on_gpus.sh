# #!/bin/bash

# # =============================================================================
# # INSTRUCTION TUNED
# # =============================================================================

# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32' > ./output/it/it_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4' > ./output/it/it_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2' > ./output/it/it_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4' > ./output/it/it_wd.out 2>&1 &

# # =============================================================================
# # NON INSTRUCTION TUNED
# # =============================================================================

# # # #!/bin/bash
# # # NO
# nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_nit.py --file test_GEMMA_no_context_not-packed.jsonl --batch_size 32'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/gemma_baseline/nit/nit_test_GEMMA_no_context.out 2>&1 &

# # # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_nit.py --file test_GEMMA_relevant_context_not-packed.jsonl --batch_size=2'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/gemma_baseline/nit/nit_test_GEMMA_rel_context.out 2>&1 &

# # # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_nit.py --file test_GEMMA_random_context_not-packed.jsonl --batch_size=2'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/gemma_baseline/nit/nit_test_GEMMA_random_context.out 2>&1 &

# # # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_nit.py --file test_GEMMA_wrong_date_context_not-packed.jsonl --batch_size=2'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/gemma_baseline/nit/nit_test_GEMMA_WD_context.out 2>&1 &


# # =============================================================================
# # TRAINED NO CONTEXT
# # =============================================================================
# NO
# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_GEMMA_no_context_not-packed.jsonl \
--batch_size 256 \
--model gemma_no_context_model.pt'  > ./output/trained_no/no_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generation_test_it.py \
--file test_GEMMA_relevant_context_not-packed.jsonl \
--batch_size 256 \
--model gemma_no_context_model.pt' > ./output/trained_no/trained_no/no_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generation_test_it.py \
--file test_GEMMA_random_context_not-packed.jsonl \
--batch_size 256 \
--model gemma_no_context_model.pt'  > ./output/trained_no/trained_no/no_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generation_test_it.py \
--file test_GEMMA_wrong_date_context_not-packed.jsonl \
--batch_size 256 \
--model gemma_no_context_model.pt'  > ./output/trained_no/trained_no/no_t_wd.out 2>&1 &


# # =============================================================================
# # TRAINED REL CONTEXT
# # =============================================================================
# # NEXT TO RUN --++--++--++

nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_no_context_not-packed.jsonl --batch_size 16 --model gemma_rel_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rel/rel_trained_test_GEMMA_no_context.out 2>&1 &

# # REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_relevant_context_not-packed.jsonl --batch_size=1 --model gemma_rel_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rel/rel_trained_test_GEMMA_rel_context.out 2>&1 &

# # RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_random_context_not-packed.jsonl --batch_size=1 --model gemma_rel_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rel/rel_trained_test_GEMMA_random_context.out 2>&1 &

# # WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_wrong_date_context_not-packed.jsonl --batch_size=1 --model gemma_rel_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rel/rel_trained_test_GEMMA_WD_context.out 2>&1 &



# =============================================================================
# TRAINED RANDOM CONTEXT
# =============================================================================
# NEXT TO RUN --++--++--++

# nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_no_context_not-packed.jsonl --batch_size 16 --model gemma_random_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rand/random_trained_test_GEMMA_no_context.out 2>&1 &

# # # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_relevant_context_not-packed.jsonl --batch_size=1 --model gemma_random_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rand/random_trained_test_GEMMA_rel_context.out 2>&1 &

# # # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_random_context_not-packed.jsonl --batch_size=1 --model gemma_random_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rand/random_trained_test_GEMMA_random_context.out 2>&1 &

# # # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/generation_test_it.py --file test_GEMMA_wrong_date_context_not-packed.jsonl --batch_size=1 --model gemma_random_context_model.pt'  > /home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/data/generation_output/trained_rand/random_trained_test_GEMMA_WD_context.out 2>&1 &
