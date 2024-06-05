#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from peft import PeftModel
import pandas as pd
from datasets import Dataset
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from huggingface_hub import login

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
    parser.add_argument('--model_path', type=str, required=True, help='hub path of  model')

    return parser.parse_args()

#endregion
#region # FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================
def format_instruction_test_nit(df, context=False):
    tokenized_instructions = []
 
    if context:
        for question, context in zip(df['question'], df[context]):

            message = f'<bos>{question} Here is the context: {context} The answer is: '

            # apply eos to end of the string
            tokenized_instructions.append(message)

    else:
        for question in df['question']:
            
            message = f'<bos>{question} The answer is: '

            # apply eos to end of the string
            tokenized_instructions.append(message)

    return tokenized_instructions


def format_instruction_test_it(df, model_path, context=False):
    tokenized_instructions = []
 
    ft_tokenizer = AutoTokenizer.from_pretrained(f"google/{model_path}", add_bos_token=True, add_eos_token=False)
    ft_tokenizer.padding_side = 'left'

    if context:
        for question, context in zip(df['question'], df[context]):
            '''
            message = [
                {"role": "user", "content": f'{question} Here is the context: {context}'},
                # {"role": "assistant", "content": answer},
            ]

            encodeds = ft_tokenizer.apply_chat_template(message, return_tensors="pt")
            output = ft_tokenizer.batch_decode(encodeds, special_tokens=True) 
            tokenized_instructions.append(output[0] + '<start_of_turn>model\nThe answer is: ')
            '''
            message = f"<start_of_turn>user\nQUESTION: {question} Here is the context: {context} <end_of_turn>\n<start_of_turn>model\nThe answer is: "
            tokenized_instructions.append(message)

    else:
        for question in df['question']:
            message = [
                {"role": "user", "content": question},
                # {"role": "assistant", "content": answer},
            ]

            encodeds = ft_tokenizer.apply_chat_template(message, return_tensors="pt")
            output = ft_tokenizer.batch_decode(encodeds, special_tokens=True) 
            tokenized_instructions.append(output[0] + '<start_of_turn>model\nThe answer is: ')

    return tokenized_instructions

def other_preprocessing(df, tokenizer, context):
    # convert from pandas to dataset obj
    dataset = Dataset.from_pandas(df)
    # TOKENIZE DATASET
    dataset = dataset.map(lambda samples: tokenizer(samples[f"{context.split('_')[0]}_prompt"]), batched=True)

    return dataset

#endregion
#region # MAIN
# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main(): # @$@
    # Set the working directory
    os.chdir('/home/dan/mini_temporal')

    # SET UP TORCH / CUDA
    # TORCH AND LOGGING SET UP
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    # Use the get_device function to set the device
    device = 'cuda' # @$@
    print(f"Using {torch.cuda.get_device_name(device)}") # @$@

    # ARGPARSE
    args = parse_args() # @$@

    #region # LOAD TEST DATA
    # =============================================================================
    # LOAD + PREPROCESS TEST DATA
    # =============================================================================
    # LOAD FILE
    print('loading test data') # @$@
    test = pd.read_json(f'./data/{args.dataset}/test.jsonl', lines=True)#, orient='records') # @$@
    # test = pd.read_json(f'./data/TQE/test.jsonl', lines=True)#, orient='records')

    # PREFIX STRINGS

    # FORMAT CONTEXT AS PROMPT
    print('formatting instructions')

    if '-it' in args.model_path: # MODEL IS INSTRUCTION TUNED
        test['no_prompt'] = format_instruction_test_it(test, model_path = args.model_path) 
        test['relevant_prompt'] = format_instruction_test_it(test, model_path = args.model_path, context = 'relevant_context') # @$@
        test['random_prompt'] = format_instruction_test_it(test, model_path = args.model_path, context = 'random_context') # @$@
        test['wrong_prompt'] = format_instruction_test_it(test, model_path = args.model_path, context = 'wrong_date_context') # @$@

    else: # MODEL IS BASE NON INSTRUCTION TUNED
        test['no_prompt'] = format_instruction_test_nit(test)
        test['relevant_prompt'] = format_instruction_test_nit(test, context = 'relevant_context') # @$@
        test['random_prompt'] = format_instruction_test_nit(test, context = 'random_context') # @$@
        test['wrong_prompt'] = format_instruction_test_nit(test, context = 'wrong_date_context') # @$@

    # TOKENIZATION
    # print('tokenizing data')
    token = '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'
    with open('./training/token.txt', 'r') as file:
        token = file.read().strip()

    login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(f"google/{args.model_path}", add_bos_token=True, add_eos_token=False, token =token)
    tokenizer.padding_side = 'left'

    # TURN INTO DATASET, SHUFFLE AND TOKENIZE DATA
    print('other preprocessing')
    test_data = other_preprocessing(test, tokenizer, args.context)

    # print('tokenization')
    inputs = tokenizer(list(test_data[f"{args.context.split('_')[0]}_prompt"]), return_tensors="pt", max_length=1800, padding=True, truncation=True)
    # Create a TensorDataset and DataLoader for manageable batch processing
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    generated_text = tokenizer.decode(dataset[0][0], skip_special_tokens=False)
    print('\n\n',f'GENERATED TEXT: {generated_text}', '\n\n')
    loader = DataLoader(dataset, batch_size=args.batch_size)  # Adjust batch size based on your GPU capacity

    all_decoded_responses = [] 
    #endregion

    #region # SET UP MODEL
    # =============================================================================
    # SET UP MODEL
    # =============================================================================
    # Specify the model path

    model_path = f"{args.model_path}"

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(f"google/{model_path}").to(device)

    # Load the adapter model
    # base_model = PeftModel.from_pretrained(base_model, model_path).to(device)

    #endregion
    #region # GENERATION
    # =============================================================================
    # GENERATION
    # =============================================================================
    for i, batch in enumerate(loader):
        input_ids, attention_mask = [b.to(device) for b in batch]
        model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        generated_ids = base_model.generate(
            **model_inputs,
            max_new_tokens=16, # @$@ 50
            do_sample=True, 
            #top_k=50, 
            temperature=.8, 
            repetition_penalty=1.15, # @$@ 1.15
            num_beams=2,
            top_p=0.99,
            top_k=0,
            length_penalty=.1
            )
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, )
        if '-it' in args.model_path:
            for j, item in enumerate(decoded_responses):
                idx = i * len(decoded_responses) + j
                question = item.split('\nmodel\n')[0][5:]
                answer = item.split("\nmodel\n")[1]

                print(json.dumps({'INDEX': idx,'QUESTION': question,'PREDICTION': answer}))
                sys.stdout.flush()
        else: # Model is not instruction tuned
            for j, item in enumerate(decoded_responses):
                idx = i * len(decoded_responses) + j
                question = item.split('The answer is: ')[0]
                answer = 'The answer is: ' + item.split('The answer is: ')[1]

                print(json.dumps({'INDEX': idx,'QUESTION': question,'PREDICTION': answer}))


        # Free up memory
        del input_ids, attention_mask, generated_ids, decoded_responses
        torch.cuda.empty_cache()  # Use cautiously
    #endregion

if __name__ == "__main__":  # @$@
    main() # @$@
