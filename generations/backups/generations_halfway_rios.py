#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from datasets import Dataset
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import json

#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

    parser.add_argument('--dataset', type=str, required=True, choices=['AQA','TQE'], help='Do you want to use AQA dataset or TQE')
    parser.add_argument('--context', type=str, required=True, choices=['no_context','random_context','relevant_context','wrong_date_context'])
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='How many pass at model at once?')
    parser.add_argument('--model_path', type=str, required=True, help='dir of save model')

    return parser.parse_args()

#endregion
#region # FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================
def format_instruction_test(df, context=False):
    tokenized_instructions = []

    # set up tokenizer
    #ft_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", add_bos_token=True, add_eos_token=True)
    ft_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", add_bos_token=False, add_eos_token=False) # RIOS: You can't use eos at test time, it will assume you are done generating and generate garbage.
    ft_tokenizer.padding_side = 'right'
    if context:
        '''
        for question, context in zip(df['question'], df[context]):
            message = [
                {"role": "user", "content": f'{question} Here is the context: {context}'},
                # {"role": "assistant", "content": answer},
            ]

            encodeds = ft_tokenizer.apply_chat_template(message, return_tensors="pt")
            output = ft_tokenizer.batch_decode(encodeds, special_tokens=True) 
            tokenized_instructions.append(output[0] + '<start_of_turn>user\n')
        '''
        for question, context in zip(df['question'], df[context]):
            prompt = f"<start_of_turn>user\n{question} Here is the context: {context}<end_of_turn>\n<start_of_turn>model\nThe correct answer is"
            tokenized_instructions.append(prompt)

    else:
        for question in df['question']:
            '''
            message = [
                {"role": "user", "content": question},
                # {"role": "assistant", "content": answer},
            ]

            #encodeds = ft_tokenizer.apply_chat_template(message, return_tensors="pt")
            '''
            prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\nThe correct answer is"
            tokenized_instructions.append(prompt)
            #output = ft_tokenizer.batch_decode(encodeds, special_tokens=True) 
            #tokenized_instructions.append(output[0] + '<start_of_turn>user\n')

    return tokenized_instructions

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

#endregion
#region # MAIN
# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main(): 
    # Set the working directory
    os.chdir('/home/dan/mini_temporal')

    # SET UP TORCH / CUDA
    # TORCH AND LOGGING SET UP
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    # Use the get_device function to set the device
    device = 'cuda'
    print(f"Using {torch.cuda.get_device_name(device)}")

    # ARGPARSE
    args = parse_args()

    #region # LOAD TEST DATA
    # =============================================================================
    # LOAD + PREPROCESS TEST DATA
    # =============================================================================
    # LOAD FILE
    test = pd.read_json(f'./data/{args.dataset}/test.jsonl', lines=True, orient='records')
    
    # PREFIX STRINGS

    # FORMAT CONTEXT AS PROMPT
    test['no_prompt'] = format_instruction_test(test)
    test['rel_prompt'] = format_instruction_test(test, 'relevant_context')
    test['rand_prompt'] = format_instruction_test(test, 'random_context')
    test['wd_prompt'] = format_instruction_test(test, 'wrong_date_context')

    # TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained(f"./training/models/{args.model_path}", add_bos_token=True, add_eos_token=False)
    # Ensure padding side is set to 'right' to avoid potential overflow issues
    tokenizer.padding_side = 'left'

    # TURN INTO DATASET, SHUFFLE AND TOKENIZE DATA
    test_data = other_preprocessing(test, tokenizer, args.context)

    inputs = tokenizer(list(test_data['no_prompt']), return_tensors="pt", max_length=1800, padding=True, truncation=True)
    # Create a TensorDataset and DataLoader for manageable batch processing
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    print(dataset[0])
    generated_text = tokenizer.decode(dataset[0][0], skip_special_tokens=False)
    print(generated_text)
    loader = DataLoader(dataset, batch_size=args.batch_size)  # Adjust batch size based on your GPU capacity

    all_decoded_responses = [] 
    #endregion

    #region # SET UP MODEL
    # =============================================================================
    # SET UP MODEL
    # =============================================================================
    # Specify the model path

    model_path = f"./training/models/{args.model_path}"

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Load the adapter model
    adapter_model = PeftModel.from_pretrained(base_model, model_path).to(device)

    #endregion
    #region # GENERATION
    # =============================================================================
    # GENERATION
    # =============================================================================
    for i, batch in enumerate(loader):
        input_ids, attention_mask = [b.to(device) for b in batch]
        model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        generated_ids = adapter_model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=True, 
            top_k=50, 
            temperature=.01, 
            repetition_penalty=2.5, 
            length_penalty=1.0
            )
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for j, item in enumerate(decoded_responses):
            idx = i * len(decoded_responses) + j
            question = item.split('user\n')[1].split('model\n')[0].strip()
            answer = item.split('model\n')[1].strip()
            #print(item)

            print(json.dumps({'INDEX': idx,'QUESTION': question,'PREDICTION': answer}))

        # Free up memory
        del input_ids, attention_mask, generated_ids, decoded_responses
        torch.cuda.empty_cache()  # Use cautiously
    #endregion

if __name__ == "__main__":  
    main()
