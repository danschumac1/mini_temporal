# #!/bin/bash

# # =============================================================================
# # IN CLASS EXAMPLE
# # =============================================================================
sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file IN_CLASS_EXAMPLE.jsonl \
--batch_size 4 \
--model mini_no_context_model.pt'

sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file IN_CLASS_EXAMPLE.jsonl \
--batch_size 4 \
--model mini_rel_context_model.pt'

sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file IN_CLASS_EXAMPLE.jsonl \
--batch_size 4 \
--model mini_random_context_model.pt'

sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file IN_CLASS_EXAMPLE.jsonl \
--batch_size 4 \
--model mini_wd_context_model.pt'

# # =============================================================================
# # INSTRUCTION TUNED
# # =============================================================================

# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32' 
#> ./output/it/it_no.out 2>&1 &

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

# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32' > ./output/nit/nit_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4' > ./output/nit/nit_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2' > ./output/nit/nit_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4' > ./output/nit/nit_wd.out 2>&1 &
# # =============================================================================
# # TRAINED NO CONTEXT
# # =============================================================================
# NO

nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model mini_no_context_model.pt'  > ./output/trained_no/no_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model mini_no_context_model.pt' > ./output/trained_no/no_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model mini_no_context_model.pt'  > ./output/trained_no/no_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model mini_no_context_model.pt'  > ./output/trained_no/no_t_wd.out 2>&1 &


# # =============================================================================
# # TRAINED REL CONTEXT
# # =============================================================================
# # NEXT TO RUN --++--++--++

nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model mini_rel_context_model.pt'  > ./output/trained_rel/rel_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model mini_rel_context_model.pt' > ./output/trained_rel/rel_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model mini_rel_context_model.pt'  > ./output/trained_rel/rel_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model mini_rel_context_model.pt'  > ./output/trained_rel/rel_t_wd.out 2>&1 &


# =============================================================================
# TRAINED RANDOM CONTEXT
# =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model mini_random_context_model.pt'  > ./output/trained_rand/rand_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model mini_random_context_model.pt' > ./output/trained_rand/rand_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model mini_random_context_model.pt'  > ./output/trained_rand/rand_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \

--model mini_random_context_model.pt'  > ./output/trained_rand/rand_t_wd.out 2>&1 &

# =============================================================================
# TRAINED WD
# =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model mini_wd_context_model.pt'  > ./output/trained_wd/wd_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model mini_wd_context_model.pt' > ./output/trained_wd/wd_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model mini_wd_context_model.pt'  > ./output/trained_wd/wd_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model mini_wd_context_model.pt'  > ./output/trained_wd/wd_t_wd.out 2>&1 &