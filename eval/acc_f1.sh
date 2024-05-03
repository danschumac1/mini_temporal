#!/bin/bash

# MAKE SURE THAT GPT COL JSONL KEYS ARE THE SAME AS THE OTHERS!
> results.jsonl

# =============================================================================
# GPT
# =============================================================================
sh -c 'python acc_f1.py \
--sub_folder gpt_cleaned \
--file gpt_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder gpt_cleaned \
--file gpt_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder gpt_cleaned \
--file gpt_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder gpt_cleaned \
--file gpt_wd_cleaned.jsonl' >> results.jsonl 2>&1
# =============================================================================
# IT
# =============================================================================

sh -c 'python acc_f1.py \
--sub_folder it_cleaned \
--file it_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder it_cleaned \
--file it_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder it_cleaned \
--file it_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder it_cleaned \
--file it_wd_cleaned.jsonl' >> results.jsonl 2>&1

# =============================================================================
# NIT
# =============================================================================
sh -c 'python acc_f1.py \
--sub_folder nit_cleaned \
--file nit_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder nit_cleaned \
--file nit_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder nit_cleaned \
--file nit_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder nit_cleaned \
--file nit_wd_cleaned.jsonl' >> results.jsonl 2>&1


# =============================================================================
# TRAINED_NO
# =============================================================================
sh -c 'python acc_f1.py \
--sub_folder trained_no_cleaned \
--file no_t_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_no_cleaned \
--file no_t_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_no_cleaned \
--file no_t_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_no_cleaned \
--file no_t_wd_cleaned.jsonl' >> results.jsonl 2>&1


# =============================================================================
# TRAINED_REL
# =============================================================================
sh -c 'python acc_f1.py \
--sub_folder trained_rel_cleaned \
--file rel_t_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_rel_cleaned \
--file rel_t_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_rel_cleaned \
--file rel_t_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_rel_cleaned \
--file rel_t_wd_cleaned.jsonl' >> results.jsonl 2>&1


# =============================================================================
# TRAINED_RAND
# =============================================================================
sh -c 'python acc_f1.py \
--sub_folder trained_rand_cleaned \
--file rand_t_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_rand_cleaned \
--file rand_t_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_rand_cleaned \
--file rand_t_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_rand_cleaned \
--file rand_t_wd_cleaned.jsonl' >> results.jsonl 2>&1


# =============================================================================
# TRAINED_WD
# =============================================================================
sh -c 'python acc_f1.py \
--sub_folder trained_wd_cleaned \
--file wd_t_no_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_wd_cleaned \
--file wd_t_rel_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_wd_cleaned \
--file wd_t_rand_cleaned.jsonl' >> results.jsonl 2>&1

sh -c 'python acc_f1.py \
--sub_folder trained_wd_cleaned \
--file wd_t_wd_cleaned.jsonl' >> results.jsonl 2>&1


