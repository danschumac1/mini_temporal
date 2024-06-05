#!/bin/bash

# =============================================================================
# GPT
# =============================================================================
# DONE IN A DIFFERENT CODE BC IT WAS A DIF FORMAT
# =============================================================================
# IT
# =============================================================================

nohup sh -c 'python generation_corrections.py \
--generation_file it_no.out  \
--sub_folder it \
--save_name it_no_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file it_rel.out  \
--sub_folder it \
--save_name it_rel_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file it_rand.out  \
--sub_folder it \
--save_name it_rand_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file it_wd.out  \
--sub_folder it \
--save_name it_wd_cleaned' > generation_corrections_log.txt 2>&1 &

# =============================================================================
# NIT
# =============================================================================

nohup sh -c 'python generation_corrections.py \
--generation_file nit_no.out  \
--sub_folder nit \
--save_name nit_no_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file nit_rel.out  \
--sub_folder nit \
--save_name nit_rel_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file nit_rand.out  \
--sub_folder nit \
--save_name nit_ramd_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file nit_wd.out  \
--sub_folder nit \
--save_name nit_wd_cleaned' > generation_corrections_log.txt 2>&1 &

# =============================================================================
# No trained
# =============================================================================
nohup sh -c 'python generation_corrections.py \
--generation_file no_t_no.out  \
--sub_folder no \
--save_name no_t_no_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file no_t_rel.out  \
--sub_folder no \
--save_name no_t_rel_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file no_t_rand.out  \
--sub_folder no \
--save_name no_t_rand_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file no_t_wd.out  \
--sub_folder no \
--save_name no_t_wd_cleaned' > generation_corrections_log.txt 2>&1 &

# =============================================================================
# rel trained
# =============================================================================
nohup sh -c 'python generation_corrections.py \
--generation_file rel_t_no.out  \
--sub_folder rel \
--save_name rel_t_no_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file rel_t_rel.out  \
--sub_folder rel \
--save_name rel_t_rel_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file rel_t_rand.out  \
--sub_folder rel \
--save_name rel_t_rand_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file rel_t_wd.out  \
--sub_folder rel \
--save_name rel_t_wd_cleaned' > generation_corrections_log.txt 2>&1 &

# =============================================================================
# rand trained
# =============================================================================
nohup sh -c 'python generation_corrections.py \
--generation_file rand_t_no.out  \
--sub_folder rand \
--save_name rand_t_no_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file rand_t_rel.out  \
--sub_folder rand \
--save_name rand_t_rel_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file rand_t_rand.out  \
--sub_folder rand \
--save_name rand_t_rand_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file rand_t_wd.out  \
--sub_folder rand \
--save_name rand_t_wd_cleaned' > generation_corrections_log.txt 2>&1 &
# =============================================================================
# wd trained
# =============================================================================
nohup sh -c 'python generation_corrections.py \
--generation_file wd_t_no.out  \
--sub_folder wd \
--save_name wd_t_no_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file wd_t_rel.out  \
--sub_folder wd \
--save_name wd_t_rel_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file wd_t_rand.out  \
--sub_folder wd \
--save_name wd_t_rand_cleaned' > generation_corrections_log.txt 2>&1 &

nohup sh -c 'python generation_corrections.py \
--generation_file wd_t_wd.out  \
--sub_folder wd \
--save_name wd_t_wd_cleaned' > generation_corrections_log.txt 2>&1 &