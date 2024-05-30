"""
Created on 05/14/2024

@author: Dan Schumacher
This is a code along from the good people at 
https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-gemma-0444d46d821c
"""

#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
import os
os.chdir('/home/dan/mini_temporal')

import json
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer
import transformers

# LOG INTO HUGGING FACE
with open('./training/token.txt', 'r') as file:
    token = file.read().strip()
login(token=token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True, # Example setting, adjust as needed
)

#endregion
#region # LOAD MODEL, TOKENIZER, AND DATASET
# =============================================================================
# LOAD MODEL, TOKENIZER, AND DATASET
# =============================================================================

# LOAD MODEL
model_id = "google/gemma-2b" # @$@ eventually set this as a CLA
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_cache=False)

# SET UP TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# LOAD DATA
train = pd.read_json('./data/train.jsonl', lines=True)
dev = pd.read_json('./data/dev.jsonl', lines=True)

#endregion
#region # CREATE PROMPTS
# =============================================================================
# CREATE PROMPTS
# =============================================================================

def add_all_prompts(df):

    no_context_prefix_text = 'Given a question, answer the question to the best of your abilities.'
    context_prefix_text = 'Given a context paragraph and a question, answer the question to the best of your abilities using the context.'

    def generate_no_context_prompt(df, prefix_text):
        user_SOS_GEMMA = '<start_of_turn>user'
        model_SOS_GEMMA = '<start_of_turn>model'
        EOS_GEMMA = '<end_of_turn>'

        GEMMA_no_context = []
        for q in df['question']:
            example_GEMMA = f"{user_SOS_GEMMA}\n{prefix_text}\n\nQUESTION: {q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
            GEMMA_no_context.append(example_GEMMA)
        return GEMMA_no_context

    def generate_context_prompt(df, prefix_text, context): 
        user_SOS_GEMMA = '<start_of_turn>user'
        model_SOS_GEMMA = '<start_of_turn>model'
        EOS_GEMMA = '<end_of_turn>'

        GEMMA_no_context = []
        for c, q in zip(df[context], df['question']):
            example_GEMMA = f"{user_SOS_GEMMA}\n{prefix_text}\n\nCONTEXT: {c}\n\nQUESTION: {q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
            GEMMA_no_context.append(example_GEMMA)
        return GEMMA_no_context


    df['no_prompt'] = generate_no_context_prompt(df, no_context_prefix_text)
    df['rel_prompt'] = generate_context_prompt(df, context_prefix_text, 'relevant_context')
    df['rand_prompt'] = generate_context_prompt(df, context_prefix_text, 'random_context')
    df['wd_prompt'] = generate_context_prompt(df, context_prefix_text, 'wrong_date_context')

add_all_prompts(train)
add_all_prompts(dev)
#endregion
#region # OTHER PREPROCESSING
# =============================================================================
# OTHER PREPROCESSING
# =============================================================================
def other_preprocessing(df):
    # convert from pandas to dataset obj
    dataset = Dataset.from_pandas(df)

    # shuffle
    dataset = dataset.shuffle(seed=1234)

    # TOKENIZE DATASET
    dataset = dataset.map(lambda samples: tokenizer(samples["no_prompt"]), batched=True) # @$@ CLA

    return dataset

train_data = other_preprocessing(train)
dev_data = other_preprocessing(dev)

#endregion
#region # ADVANCED FORMATTING
# =============================================================================
# LORA
# =============================================================================
# lora_config = LoraConfig(
#     r = 256,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_alpha = 256,
#     lora_dropout = 0, # Dropout = 0 is currently optimized
#     bias = "none",    # Bias = “none” is currently optimized
# ) 

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
) 
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# CALCULATE THE NUMBER OF TRAINABLE PARAMS
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

#endregion
#region # SET UP TRAINER
# =============================================================================
# SET UP TRAINER
# =============================================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=dev_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        run_name = 'gemma-2b_no-trained',
        num_train_epochs= 12,
        per_device_train_batch_size= 32,
        per_device_eval_batch_size = 32,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=8,
        warmup_steps=0.03,
        gradient_checkpointing=True,
        # max_steps=100,
        learning_rate=2e-5,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        # gradient_checkpointing=True,
        # use_cache=False
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    max_seq_length=512  # Specify the desired max sequence length
)

#endregion
#region # TRAINING PROCESS
# =============================================================================
# TRAINING PROCESS
# =============================================================================
# EXICUTE THE FINE TUNING PROCESS
trainer.train()

# Save the fine-tuned model
new_model = "./training/models/nit_trained/nit_mini_no_context_model"
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
#endregion










# """
# Created on 05/14/2024

# @author: Dan Schumacher
# This is a code along from the good people at 
# https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-gemma-0444d46d821c
# """

# #endregion
# #region # IMPORTS AND SET UP
# # =============================================================================
# # IMPORTS AND SET UP
# # =============================================================================
# # SET DIRECTORY
# import os
# os.chdir('/home/dan/mini_temporal')

# # IMPORTS
# import json
# import pandas as pd
# import torch
# from datasets import Dataset, load_dataset
# from huggingface_hub import login
# from peft import LoraConfig, PeftModel
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     pipeline,
#     logging,
# )
# from trl import SFTTrainer
# import transformers

# # LOG INTO HUGGING FACE
# with open('./training/token.txt', 'r') as file:
#     token = file.read().strip()
# login(token=token)

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True, # Example setting, adjust as needed
# )

# #endregion
# #region # LOAD MODEL, TOKENIZER, AND DATASET
# # =============================================================================
# # LOAD MODEL, TOKENIZER, AND DATASET
# # =============================================================================

# # LOAD MODEL
# model_id = "google/gemma-2b" # @$@ eventually set this as a CLA
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_cache=False)

# # SET UP TOKENIZER
# tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

# # LOAD DATA
# # os.getcwd()
# train = pd.read_json('./data/train.jsonl', lines=True)
# dev = pd.read_json('./data/dev.jsonl', lines=True)

# #endregion
# #region # CREATE PROMPTS
# # =============================================================================
# # CREATE PROMPTS
# # =============================================================================
# no_context_prefix_text = 'Given a question, answer the question to the best of your abilities.'
# context_prefeix_text = 'Given a context paragraph and a question, answer the question to the best of your abilities using the context.'

# def generate_no_context_prompt(df, prefix_text):
#     user_SOS_GEMMA = '<start_of_turn>user'
#     model_SOS_GEMMA = '<start_of_turn>model'
#     EOS_GEMMA = '<end_of_turn>'

#     GEMMA_no_context = []
#     for q in df['question']:
#         example_GEMMA = f"{user_SOS_GEMMA}\n{prefix_text}\n\nQUESTION: {q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
#         GEMMA_no_context.append(example_GEMMA)
#     return GEMMA_no_context

# def generate_context_prompt(df, prefix_text, context): 
#     user_SOS_GEMMA = '<start_of_turn>user'
#     model_SOS_GEMMA = '<start_of_turn>model'
#     EOS_GEMMA = '<end_of_turn>'

#     GEMMA_no_context = []
#     for c, q in zip(df[context], df['question']):
#         example_GEMMA = f"{user_SOS_GEMMA}\n{prefix_text}\n\nCONTEXT: {c}\n\nQUESTION: {q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
#         GEMMA_no_context.append(example_GEMMA)
#     return GEMMA_no_context

# def add_all_prompts(df):
#     df['no_prompt'] = generate_no_context_prompt(df, no_context_prefix_text)
#     df['rel_prompt'] = generate_context_prompt(df, context_prefeix_text, 'relevant_context')
#     df['rand_prompt'] = generate_context_prompt(df, context_prefeix_text, 'random_context')
#     df['wd_prompt'] = generate_context_prompt(df, context_prefeix_text, 'wrong_date_context')

# add_all_prompts(train)
# add_all_prompts(dev)
# train['no_prompt'].iloc[0]
# #endregion
# #region # OTHER PREPROCESSING
# # =============================================================================
# # OTHER PREPROCESSING
# # =============================================================================

# def other_preprocessing(df):
#     # convert from pandas to dataset obj
#     dataset = Dataset.from_pandas(df)

#     # shuffle
#     dataset = dataset.shuffle(seed=1234)

#     # TOKENIZE DATASET
#     dataset = dataset.map(lambda samples: tokenizer(samples["no_prompt"]), batched=True) # @$@ CLA

#     return dataset

# train_data = other_preprocessing(train)
# dev_data = other_preprocessing(dev)

# #endregion
# #region # ADVANCED FORMATTING
# # =============================================================================
# # LORA
# # =============================================================================
# # APPLY LORA
# from peft import LoraConfig, get_peft_model
# lora_config = LoraConfig(
#     r = 256,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_alpha = 256,
#     lora_dropout = 0, # Dropout = 0 is currently optimized
#     bias = "none",    # Bias = “none” is currently optimized
# ) 
# model = get_peft_model(model, lora_config)
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()

# # CALCULATE THE NUMBER OF TRAINABLE PARAMS
# trainable, total = model.get_nb_trainable_parameters()
# print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")


# #endregion
# #region # SET UP TRAINER
# # =============================================================================
# # SET UP TRAINER
# # =============================================================================
# #new code using SFTTrainer

# tokenizer.pad_token = tokenizer.eos_token
# torch.cuda.empty_cache()

# tokenizer.padding_side = 'right'

# trainer = SFTTrainer(
    
#     model=model,
#     train_dataset=train_data,
#     eval_dataset=dev_data,
#     dataset_text_field="prompt",
#     peft_config=lora_config,
#     args=transformers.TrainingArguments(
#         run_name = 'gemma-2b_no-trained',
#         num_train_epochs= 12,
#         per_device_train_batch_size= 32,
#         per_device_eval_batch_size = 32,
#         eval_accumulation_steps=1,
#         gradient_accumulation_steps=8,
#         warmup_steps=0.03,
#         gradient_checkpointing=True,
#         # max_steps=100,
#         learning_rate=2e-4,
#         logging_steps=1,
#         output_dir="outputs",
#         optim="paged_adamw_8bit",
#         save_strategy="epoch",
#         # gradient_checkpointing=True,
#         # use_cache=False
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     max_seq_length=512  # Specify the desired max sequence length
# )

# #endregion
# #region # TRAINING PROCESS
# # =============================================================================
# # TRAINING PROCESS
# # =============================================================================
# # EXICUTE THE FINE TUNING PROCESS
# trainer.train()

# # Name of the model you will be pushing to huggingface model hub
# new_model = "./training/models/nit_trained/nit_mini_no_context_model"
# trainer.model.save_pretrained(new_model)
# tokenizer.save_pretrained(new_model)