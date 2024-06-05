"""
Created on 05/14/2024

@author: Dan Schumacher
This code takes a GEMMA model and a dataset, processes the data, fine-tunes, and saves a model for evaluation.
"""

##### EXAMPLE TO RUN SCRIPT #####
# nohup bash -c "CUDA_VISIBLE_DEVICES=0 python fine_tune_gemma_MIXED_nit.py \
#     --epochs 1 \
#     --dataset TQE \
#     --model_context mixed_context \
#     --base_model gemma-2b \
#     --batch_size 16" > test_train.out


#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
# print('IMPORTS AND SET UP')
import os
os.chdir('/home/dan/mini_temporal')

# import json
import pandas as pd
# import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # TrainingArguments,
    # pipeline,
    # logging,
)

from trl import SFTTrainer
import transformers
import argparse
import wandb 
import warnings

#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
# print('ARGPARSE')
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")
    parser.add_argument('--dataset', type=str, required=True, choices= ['AQA','TQE'], help='Do you want to use AQA dataset or TQE')
    parser.add_argument('--model_context', type=str, required=True, choices= ['no_context','random_context','relevant_context','wrong_date_context','mixed_context'], help='Model context')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='How many pass at model at once?')
    parser.add_argument('--base_model', type = str, required=True, choices= ['gemma-2b','gemma-1.1-2b-it','gemma-7b','gemma-1.1-7b-it'])
    parser.add_argument('--lr', type = float, required=False, default=2e-5, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, required=False, default=6, help = 'How many training epochs?')
    return parser.parse_args()

# print('Argument parser defined')  # @$@
# =============================================================================
# FUNCTIONS
# =============================================================================
# Some of our answers in our TQE dataset have multiple correct answers (exmp: US, USA)
# these are seperated with the __or__ characters
# we will just grab the first answer during training. 
def check_or(answer):
    if '__or__' in answer:
        answer = answer.split('__or__')[0]
    return answer

def format_instruction(df, model_path, context=False):
    tokenized_instructions = []
    # USING A BASE MODEL (NOR INSTRUCTION TUNED)
        
    if context: # BASE MODEL WITH CONTEXT
        for question, context, answer in zip(df['question'], df[context], df['answer']):

            # Make sure only 1 answer and no __or__ tokens
            answer = check_or(answer)

            message = f'<bos>{question} Here is the context: {context} The answer is: {answer}<eos>'

            # apply eos to end of the string
            tokenized_instructions.append(message)

    else: # BASE MODEL WITHOUT CONTEXT
        for question, answer in zip(df['question'], df['answer']):

            # Make sure only 1 answer and no __or__ tokens
            answer = check_or(answer)

            message = f'<bos>{question} The answer is: {answer}<eos>'

            # apply eos to end of the string
            tokenized_instructions.append(message)
            

    # during our training, lets print one to make sure it looks good.
    # print(tokenized_instructions[-1])
    return tokenized_instructions


def other_preprocessing(df, tokenizer, context):
    print(f"Preprocessing dataset with context: {context}")

    # convert from pandas to dataset obj
    dataset = Dataset.from_pandas(df)
    # shuffle
    dataset = dataset.shuffle(seed=1234)

    # TOKENIZE DATASET
    prompt = context.split('_context')[0] + '_prompt'
    print(f"Using prompt column: {prompt}")

    # Print some sample data to debug
    print(f"First 5 entries of df[{prompt}]:")
    print(df[prompt].head())

    dataset = dataset.map(lambda x: tokenizer(x[prompt]), batched=True)
    
    # Check dataset structure after tokenization
    print("Dataset after tokenization:")
    print(dataset[0])

    # Decode to check if the data is correct
    try:
        generated_text = tokenizer.decode(dataset[0][0], skip_special_tokens=False)
        print('\n\n',f'GENERATED TEXT: {generated_text}', '\n\n')
    except KeyError as e:
        print(f"Error decoding dataset: {e}")

    return dataset

def assign_mixed_context(index):
    if index % 4 == 0:
        return 'no_prompt'
    elif index % 3 == 0:
        return 'rel_prompt'
    elif index % 2 == 0:
        return 'rand_prompt'
    else:
        return 'wd_prompt'

#endregion
#region # MAIN
# =============================================================================
# MAIN
# =============================================================================

def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    # PARSE ARGUMENT
    args = parse_args() # @$@

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
    print('load model')
    # LOAD MODEL
    model_id = f"google/{args.base_model}" # @$@
    # model_id = 'google/gemma-2b'# @$@

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_cache=False)

    print('tokenizer')
    # SET UP TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=False, add_eos_token=False, )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    print('load data')
    # LOAD DATA
    train = pd.read_json(f'./data/{args.dataset}/train.jsonl', lines=True) # @$@
    dev = pd.read_json(f'./data/{args.dataset}/dev.jsonl', lines=True) # @$@
    # train = pd.read_json(f'./data/TQE/train.jsonl', lines=True) # @$@
    # dev = pd.read_json(f'./data/TQE/dev.jsonl', lines=True) # @$@
    print('Data loaded')
    #endregion
    #region # CREATE PROMPTS
    # =============================================================================
    # CREATE PROMPTS
    # =============================================================================
    train['no_prompt'] = format_instruction(train, model_path = args.base_model) # @$@
    train['rel_prompt'] = format_instruction(train, model_path = args.base_model, context = 'relevant_context') # @$@
    train['rand_prompt'] = format_instruction(train, model_path = args.base_model, context = 'random_context') # @$@
    train['wd_prompt'] = format_instruction(train, model_path = args.base_model, context = 'wrong_date_context') # @$@

    dev['no_prompt'] = format_instruction(dev, model_path = args.base_model) # @$@
    dev['rel_prompt'] = format_instruction(dev, model_path = args.base_model, context = 'relevant_context') # @$@
    dev['rand_prompt'] = format_instruction(dev, model_path = args.base_model, context = 'random_context') # @$@
    dev['wd_prompt'] = format_instruction(dev, model_path = args.base_model, context = 'wrong_date_context') # @$@

    # train['no_prompt'] = format_instruction(train, model_path = model_id) # @$@
    # train['rel_prompt'] = format_instruction(train, model_path = model_id, context = 'relevant_context') # @$@
    # train['rand_prompt'] = format_instruction(train, model_path = model_id, context = 'random_context') # @$@
    # train['wd_prompt'] = format_instruction(train, model_path = model_id, context = 'wrong_date_context') # @$@

    # dev['no_prompt'] = format_instruction(dev, model_path = model_id) # @$@
    # dev['rel_prompt'] = format_instruction(dev, model_path = model_id, context = 'relevant_context') # @$@
    # dev['rand_prompt'] = format_instruction(dev, model_path = model_id, context = 'random_context') # @$@
    # dev['wd_prompt'] = format_instruction(dev, model_path = model_id, context = 'wrong_date_context') # @$@



    # Applying the function to create 'mixed_prompt' column
    train['mixed_prompt'] = train.apply(lambda row: row[assign_mixed_context(row.name)], axis=1)
    dev['mixed_prompt'] = dev.apply(lambda row: row[assign_mixed_context(row.name)], axis=1)

    #region # OTHER PREPROCESSING
    train_data = other_preprocessing(train, tokenizer, args.model_context)
    dev_data = other_preprocessing(dev, tokenizer, args.model_context)
    print('Data preprocessed')  # @$@
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
    print('Model prepared for training')
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
            warmup_steps=1,                 # FATEMEH
            gradient_checkpointing=True,
            learning_rate=args.lr,
            logging_steps=25,               # FATEMEH
            eval_steps= 25,                 # FATEMEH
            output_dir="outputs",
            optim="paged_adamw_8bit",
            save_strategy="no",
            evaluation_strategy= 'steps',   # FATEMEH
            report_to="wandb"  # Integrate with wandb
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        max_seq_length=512  # Specify the desired max sequence length
    )
    print('Trainer set up')
    #endregion
    #region # TRAINING PROCESS
    # =============================================================================
    # TRAINING PROCESS
    # =============================================================================
    # EXECUTE THE FINE TUNING PROCESS
    trainer.train()

    # Specify save location of new model
    new_model = f"./training/models/{args.dataset}/{args.base_model.replace('.','_')}/{args.model_context}"

    # make sure that location exists
    os.makedirs(new_model, exist_ok=True)

    # save the model
    trainer.model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
    print(f"Model and tokenizer saved at {new_model}")

if __name__ == '__main__':
    main()