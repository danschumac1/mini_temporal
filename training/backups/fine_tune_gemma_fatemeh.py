"""
Created on 05/14/2024

@author: Dan Schumacher
This code takes a GEMMA model and a dataset, processes the data, fine-tunes, and saves a model for evaluation.
"""

#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
# print('IMPORTS AND SET UP')
import os
os.chdir('/home/dan/mini_temporal')

import json
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
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
import argparse
import wandb 
import warnings

# """TEST OF FORMATTING INPUT OF THE MODEL"""
# model_id = f"google/gemma-2b" 

# tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=False, add_eos_token=False, ) # @$@
# messages = [
#     {"role": "user", "content": "Who was named president of Disney-ABC television group in 2004? Here is the context: In 2004, Anne Sweeney was named president of the Disney-ABC Television Group. She became one of the most powerful women in the entertainment industry, overseeing a vast portfolio that included the ABC Television Network, Disney Channel Worldwide, ABC Family, and SOAPnet, among others. Sweeney's tenure at Disney-ABC was marked by her focus on expanding the company's digital presence and embracing new technologies to distribute content."},
#     {"role": "assistant", "content": "Anne Sweeney"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
# tokenizer.batch_decode(encodeds, special_tokens=True)


# print('Imports done')

#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
# print('ARGPARSE')
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

    parser.add_argument('--dataset', type=str, required=True, choices=['AQA','TQE'], help='Do you want to use AQA dataset or TQE')
    parser.add_argument('--model_context', type=str, required=True, choices=['no_context','random_context','relevant_context','wrong_date_context'], help='Model context')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='How many pass at model at once?')
    parser.add_argument('--base_model', type = str, required=True, choices=['gemma-2b','gemma-2b-it','gemma-7b','gemma-7b-it'])
    parser.add_argument('--lr', type = float, required=False, default=2e-5, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, required=False, default=6, help = 'How many training epochs?')

    return parser.parse_args()
# print('Argument parser defined')  # @$@
# =============================================================================
# FUNCTIONS
# =============================================================================
# print('DEFINING FUNCTIONS')

def generate_no_context_prompt(df):
    GEMMA_no_context = []
    for q, a in zip(df['question'], df['answer']):
        example_GEMMA = f"<|im_start|>user\n {q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        GEMMA_no_context.append(example_GEMMA)
    return GEMMA_no_context

def generate_context_prompt(df, context): 
    GEMMA_no_context = []
    for c, q, a in zip(df[context], df['question'], df['answer']):
        example_GEMMA = f"<|im_start|>user\n {q} Here is the context: {c}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        GEMMA_no_context.append(example_GEMMA)
    return GEMMA_no_context

def other_preprocessing(df, tokenizer, context):
    # convert from pandas to dataset obj
    dataset = Dataset.from_pandas(df)

    # shuffle
    dataset = dataset.shuffle(seed=1234)

    # TOKENIZE DATASET
    if context == 'random_context':
        context_plug = 'rand'
    elif context == 'relevant_context':
        context_plug = 'rel'
    elif context == 'wrong_date_context':
        context_plug = 'wd'
    elif context == 'no_context':
        context_plug = 'no'

    dataset = dataset.map(lambda samples: tokenizer(samples[f'{context_plug}_prompt']), batched=True)

    return dataset

def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    # PARSE ARGUMENT
    args = parse_args()

    # Initialize wandb
    print('WANDB')
    wandb.init(project="temporal_understanding", config=args)
    print('LOG INTO HUGGING FACE')
    # LOG INTO HUGGING FACE
    with open('./training/token.txt', 'r') as file:
        token = file.read().strip()
    login(token=token)

    # print('BNB CONFIG')

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False, # Example setting, adjust as needed
    )

    #endregion
    #region # LOAD MODEL, TOKENIZER, AND DATASET
    # =============================================================================
    # LOAD MODEL, TOKENIZER, AND DATASET
    # =============================================================================
    # print('load model')
    # LOAD MODEL
    model_id = f"google/{args.base_model}" 
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_cache=False)

    # print('tokenizer')
    # SET UP TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=False, add_eos_token=False, ) # <bos>blah <eos> 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # print('load data')
    # LOAD DATA
    train = pd.read_json(f'./data/{args.dataset}/train.jsonl', lines=True)
    dev = pd.read_json(f'./data/{args.dataset}/dev.jsonl', lines=True)

    # print('Data loaded')  # @$@
    #endregion
    #region # CREATE PROMPTS
    # =============================================================================
    # CREATE PROMPTS
    # =============================================================================
    # print('create prompts')
    no_context_prefix_text = 'Given a question, answer the question to the best of your abilities.'
    context_prefix_text = 'Given a context paragraph and a question, answer the question to the best of your abilities using the context.'

    train['no_prompt'] = generate_no_context_prompt(train)
    train['rel_prompt'] = generate_context_prompt(train, 'relevant_context')
    train['rand_prompt'] = generate_context_prompt(train, 'random_context')
    train['wd_prompt'] = generate_context_prompt(train, 'wrong_date_context')
    
    dev['no_prompt'] = generate_no_context_prompt(dev)
    dev['rel_prompt'] = generate_context_prompt(dev, 'relevant_context')
    dev['rand_prompt'] = generate_context_prompt(dev, 'random_context')
    dev['wd_prompt'] = generate_context_prompt(dev, 'wrong_date_context')
    # print('Prompts created')  # @$@
    #endregion
    #region # OTHER PREPROCESSING
    train_data = other_preprocessing(train, tokenizer, args.model_context)
    dev_data = other_preprocessing(dev, tokenizer, args.model_context)
    # print('Data preprocessed')  # @$@
    #endregion
    #region # ADVANCED FORMATTING
    # =============================================================================
    # LORA
    # =============================================================================
    print('lora')
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
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
    # print('Model prepared for training')
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
            run_name = f'{args.base_model}-{args.model_context}-trained',
            num_train_epochs= args.epochs,
            per_device_train_batch_size= args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=8,
            warmup_steps=0.03,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            report_to="wandb",               # Integrate with wandb
            # logging_steps=5,              # When to start reporting loss
            # logging_dir="./logs",        # Directory for storing logs
            # save_strategy="steps",       # Save the model checkpoint every logging step
            # save_steps=25,               # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            # eval_steps=25,               # Evaluate and save checkpoints every 50 steps
            # do_eval=True,                # Perform evaluation at the end of training
            # report_to="wandb",           # Comment this out if you don't want to use weights & baises
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

            # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            # max_seq_length=512  # Specify the desired max sequence length
        ),



    )
    # print('Trainer set up')
    #endregion
    #region # TRAINING PROCESS
    # =============================================================================
    # TRAINING PROCESS
    # =============================================================================
    # EXECUTE THE FINE TUNING PROCESS
    # print('Starting training')  # @$@
    trainer.train()
    # print('Training completed')  # @$
    # Specify save location of new model
    new_model = f"./training/models/{args.base_model}/{args.model_context}"

    # make sure that location exists
    os.makedirs(new_model, exist_ok=True)

    # save the model
    trainer.model.save_pretrained
    print(f"Model and tokenizer saved at {new_model}")

if __name__ == '__main__':
    main()

# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python fine_tune_gemma_rios.py \
# --dataset TQE \
# --model_context no_context \
# --base_model gemma-2b' > c0_run_training_log.txt 2>&1 &