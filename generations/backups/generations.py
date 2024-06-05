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
    no_context_prefix_text = 'Given a question, answer the question to the best of your abilities.'
    context_prefix_text = 'Given a context paragraph and a question, answer the question to the best of your abilities using the context.'

    # FORMAT CONTEXT AS PROMPT
    test['no_prompt'] = generate_no_context_prompt(test, no_context_prefix_text)
    test['rel_prompt'] = generate_context_prompt(test, context_prefix_text, 'relevant_context')
    test['rand_prompt'] = generate_context_prompt(test, context_prefix_text, 'random_context')
    test['wd_prompt'] = generate_context_prompt(test, context_prefix_text, 'wrong_date_context')

    # TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained(f"./training/models/{args.model_path}", add_bos_token=True, add_eos_token=False)
    # Ensure padding side is set to 'right' to avoid potential overflow issues
    tokenizer.padding_side = 'right'

    # TURN INTO DATASET, SHUFFLE AND TOKENIZE DATA
    test_data = other_preprocessing(test, tokenizer, args.context)

    inputs = tokenizer(list(test_data['no_prompt']), return_tensors="pt", max_length=1800, padding=True, truncation=True)
    # Create a TensorDataset and DataLoader for manageable batch processing
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
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
            question1 = item.split('\n\nQUESTION: ')[1].split('\nmodel')[0]
            answer = item.split('\nmodel')[1]

            print(json.dumps({'INDEX': idx,'QUESTION': question1,'PREDICTION': answer}))

        # Free up memory
        del input_ids, attention_mask, generated_ids, decoded_responses
        torch.cuda.empty_cache()  # Use cautiously
    #endregion

if __name__ == "__main__":  
    main()


# #endregion
# #region # IMPORTS AND SET UP
# # =============================================================================
# # IMPORTS AND SET UP
# # =============================================================================

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import pandas as pd
# from datasets import Dataset
# import os
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import logging
# import json

# # Set the working directory
# os.chdir('/home/dan/mini_temporal')


# # SET UP TORCH / CUDA
# # TORCH AND LOGGING SET UP
# torch.cuda.empty_cache()
# logging.basicConfig(level=logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("torch").setLevel(logging.ERROR)

# # Use the get_device function to set the device
# device = 'cuda'
# print(f"Using {torch.cuda.get_device_name(device)}")

# #endregion
# #region # ARGPARSE
# # =============================================================================
# # ARGPARSE
# # =============================================================================
# import argparse

# # print('ARGPARSE')
# def parse_args():
#     parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

#     parser.add_argument('--dataset', type=str, required=True, choices=['AQA','TQE'], help='Do you want to use AQA dataset or TQE')
#     parser.add_argument('--context', type=str, required=True, choices=['no_context','random_context','relevant_context','wrong_date_context'])
#     parser.add_argument('--batch_size', type=int, required=False, default=32, help='How many pass at model at once?')
#     parser.add_argument('--model_path', type = str, required=True, help = 'dir of save model')
#     # parser.add_argument('--lr', type = float, required=False, default=2e-5, help = 'Learning rate')
#     # parser.add_argument('--epochs', type = int, required=False, default=6, help = 'How many training epochs?')

#     return parser.parse_args()

# args = parse_args

# #endregion
# #region # FUNCTIONS
# # =============================================================================
# # FUNCTIONS
# # =============================================================================
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

# def other_preprocessing(df, tokenizer, context):
#     # convert from pandas to dataset obj
#     dataset = Dataset.from_pandas(df)

#     # shuffle
#     dataset = dataset.shuffle(seed=1234)

#     # TOKENIZE DATASET
#     if context == 'random_context':
#         context_plug = 'rand'
#     elif context == 'relevant_context':
#         context_plug = 'rel'
#     elif context == 'wrong_date_context':
#         context_plug = 'wd'
#     elif context == 'no_context':
#         context_plug = 'no'

#     dataset = dataset.map(lambda samples: tokenizer(samples[f'{context_plug}_prompt']), batched=True)

#     return dataset

# #endregion
# #region # LOAD TEST DATA
# # =============================================================================
# # LOAD + PREPROCESS TEST DATA
# # =============================================================================
# # LOAD FILE
# test = pd.read_json(f'./data/{args.dataset}/test.jsonl', lines=True, orient='records')

# # PREFIX STRINGS
# no_context_prefix_text = 'Given a question, answer the question to the best of your abilities.'
# context_prefix_text = 'Given a context paragraph and a question, answer the question to the best of your abilities using the context.'

# # FORMAT CONTEXT AS PROMPT
# test['no_prompt'] = generate_no_context_prompt(test, no_context_prefix_text)
# test['rel_prompt'] = generate_context_prompt(test, context_prefix_text, 'relevant_context')
# test['rand_prompt'] = generate_context_prompt(test, context_prefix_text, 'random_context')
# test['wd_prompt'] = generate_context_prompt(test, context_prefix_text, 'wrong_date_context')

# # TOKENIZATION
# tokenizer = AutoTokenizer.from_pretrained("./training/models/nit_trained/nit_mini_no_context_model_rios")
# # Ensure padding side is set to 'right' to avoid potential overflow issues
# tokenizer.padding_side = 'right'

# # TURN INTO DATSET, SHUFFLE AND TOKENIZE DATA
# test_data = other_preprocessing(test, tokenizer, args.context)

# inputs = tokenizer(list(test_data['no_prompt']), return_tensors="pt", max_length=1800, padding=True, truncation=True)
# # Create a TensorDataset and DataLoader for manageable batch processing
# dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
# loader = DataLoader(dataset, batch_size=args.batch_size)  # Adjust batch size based on your GPU capacity

# all_decoded_responses = [] 
# #endregion
# #region # SET UP MODEL
# # =============================================================================
# # SET UP MODEL
# # =============================================================================
# # Specify the model path

# model_path = f"./training/models/{args.model_path}"

# # Load the base model
# base_model = AutoModelForCausalLM.from_pretrained(model_path)

# # Load the adapter model
# adapter_model = PeftModel.from_pretrained(base_model, model_path).to(device)

# for i, batch in enumerate(loader):
#     input_ids, attention_mask = [b.to(device) for b in batch]
#     model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
#     model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
#     generated_ids = adapter_model.generate(
#         **model_inputs,
#         max_new_tokens=50,
#         do_sample=True, 
#         top_k=50, 
#         temperature=.01, 
#         repetition_penalty=2.5, 
#         length_penalty=1.0
#         )
#     decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

#     for j, item in enumerate(decoded_responses):
#         answer = item.split('\nmodel')[1]
#         question = item[5:].split('\nmodel')[0]
#         print(json.dumps({'INDEX': i * len(decoded_responses) + j, 'answer':answer, 'OUTPUT': item}))

#     # Free up memory
#     del input_ids, attention_mask, generated_ids, decoded_responses
#     torch.cuda.empty_cache()  # Use cautiously

# # # Example text generation function
# # def generate_text(prompt, model, tokenizer, max_length=50):
# #     inputs = tokenizer(prompt, return_tensors="pt")
# #     input_ids = inputs["input_ids"]

# #     generated_ids = model.generate(
# #         input_ids,
# #         max_new_tokens=50,
# #         do_sample=True, 
# #         top_k=50, 
# #         temperature=.01, 
# #         repetition_penalty=2.5, 
# #         length_penalty=1.0
# #         )
    
# #     generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# #     return generated_text

# # # Test the function with a prompt
# # prompt = test_data[2]['rel_prompt']
# # prompt
# # # prompt = """Who was named president of Disney-ABC television group in 2004?"""

# # generated_text = generate_text(prompt, adapter_model, tokenizer).split('model\n')
# # print(generated_text)

# # test_data[0]['answer']
