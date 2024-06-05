#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
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
 
    if context:
        for question, context in zip(df['question'], df[context]):

            message = f'QUESTION: {question} Here is the context: {context} The answer is: '
            #message = f"<start_of_turn>user\nQUESTION: {question} Here is the context: {context} <end_of_turn>\n<start_of_turn>model\nThe answer is"

            # apply eos to end of the string
            tokenized_instructions.append(message)

    else:
        for question in df['question']:
            
            message = f'{question} The answer is: '

            # apply eos to end of the string
            tokenized_instructions.append(message)

    return tokenized_instructions

def other_preprocessing(df, tokenizer, context):
    # convert from pandas to dataset obj
    dataset = Dataset.from_pandas(df)

    # shuffle
    dataset = dataset.shuffle(seed=1234)

    # TOKENIZE DATASET
    dataset = dataset.map(lambda samples: tokenizer(samples[f"{context.split('_')[0]}_prompt"]), batched=False)

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
    # print('loading test data')
    test = pd.read_json(f'./data/{args.dataset}/test.jsonl', lines=True, orient='records')
    
    # PREFIX STRINGS

    # FORMAT CONTEXT AS PROMPT
    # print('formatting instructions')
    test['no_prompt'] = format_instruction_test(test)
    test['relevant_prompt'] = format_instruction_test(test, context = 'relevant_context')
    test['random_prompt'] = format_instruction_test(test, context = 'random_context')
    test['wrong_prompt'] = format_instruction_test(test, context = 'wrong_date_context')

    # TOKENIZATION
    # print('tokenizing data')
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}", add_bos_token=True, add_eos_token=False)
    tokenizer.padding_side = 'left'

    # TURN INTO DATASET, SHUFFLE AND TOKENIZE DATA
    # print('other preprocessing')
    #test_data = other_preprocessing(test, tokenizer, args.context)

    # print('tokenization')
    #inputs = tokenizer(list(test_data[f"{args.context.split('_')[0]}_prompt"]), return_tensors="pt", max_length=1800, padding=False, truncation=False)
    # Create a TensorDataset and DataLoader for manageable batch processing
    #dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    #generated_text = tokenizer.decode(dataset[0][0], skip_special_tokens=False)
    #print('\n\n',f'GENERATED TEXT: {generated_text}', '\n\n')
    #loader = DataLoader(dataset, batch_size=args.batch_size)  # Adjust batch size based on your GPU capacity

    all_decoded_responses = [] 
    #endregion

    #region # SET UP MODEL
    # =============================================================================
    # SET UP MODEL
    # =============================================================================
    # Specify the model path

    model_path = f"{args.model_path}"

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Load the adapter model
    adapter_model = PeftModel.from_pretrained(base_model, model_path).to(device)

    #endregion
    #region # GENERATION
    # =============================================================================
    # GENERATION
    # =============================================================================
    for i, prompt in enumerate(test['relevant_prompt']):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generated_ids = adapter_model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=True, 
            top_k=50, 
            temperature=.8, 
            num_beams=2,
            repetition_penalty=1.15, 
            length_penalty=.1
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        #print(generated_text)

        '''
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
            length_penalty=.3
            )
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for j, item in enumerate(decoded_responses):
            idx = i * len(decoded_responses) + j
            print('OUT')
            print(item)
            print('dOUT')
            sys.stdout.flush()

        # Free up memory
        '''
        question = generated_text.split('The answer is: ')[0]
        answer = 'The answer is: ' + generated_text.split('The answer is: ')[1]
        print(json.dumps({'INDEX': i,'QUESTION': question,'PREDICTION': answer}))
        #del input_ids, attention_mask, generated_ids, decoded_responses
        del input_ids
        sys.stdout.flush()
        torch.cuda.empty_cache()  # Use cautiously
    #endregion

if __name__ == "__main__":  
    main()
