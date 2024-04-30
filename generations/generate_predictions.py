import os
# AN ERROR TOLD ME TO DO THIS?
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"




import sys
import torch
import pandas as pd
import json
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set working directory
os.chdir('/home/dan/DeepLearning/mini_temporal')

# Disable certain PyTorch optimizations
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# COMMAND-LINE ARGUMENTS
def parse_args():
    #description
    parser = argparse.ArgumentParser(description="Generating QA answers from Gemma models")
    # args
    parser.add_argument('--file', type=str, default='test_GEMMA_no_context.jsonl', help='dataset file') # @$@ change this
    parser.add_argument('--model', type=str, default='base', help='base if using non-instruction-tuned, else path to model.pt') # @$@ change this (eventually)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')  # New batch size argument
    # pass back
    return parser.parse_args()

# initiate
args = parse_args()

# TORCH AND LOGGING SET UP
torch.cuda.empty_cache()
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Use the get_device function to set the device
device = 'cuda'
print(f"Using {torch.cuda.get_device_name(device)}")

# LOAD DATASET AND TOKEN
dataset_folder = './data/final/test/'
dataset = pd.read_json(f'{dataset_folder}/{args.file}', lines=True)


with open('./generations/token.txt', 'r') as file:
    token = file.read().strip()

# SET UP MODEL AND TOKENIZER
if args.model == 'base': # out of the box
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b-it",
            torch_dtype=torch.bfloat16,
        ).to(device)
    except Exception as e:
        print(f"Failed to load base model: {str(e)}")
        sys.exit(1)
else: # one of our fine tuned models saved as a .pt file
    try:
        model_folder = '/home/dan/DeepLearning/TemporalUnderstandingInLLMs/temporalUnderstanding_repo/models'
        model_path = f'{model_folder}/{args.model}'
        model = torch.load(model_path, map_location=device,).to(device)
    except Exception as e:
        print(f"Failed to load model from {model_path}: {str(e)}")
        sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=token, padding_side = 'left',  truncation_side= 'left')
tokenizer.pad_token = tokenizer.eos_token

# BATCHING 
max_seq_length = 1800 # find what is your max
inputs = tokenizer(list(dataset['prompt']), return_tensors="pt", max_length=max_seq_length, padding=True, truncation=True)
# Create a TensorDataset and DataLoader for manageable batch processing
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
loader = DataLoader(dataset, batch_size=args.batch_size)  # Adjust batch size based on your GPU capacity

all_decoded_responses = []

for i, batch in enumerate(loader):
    input_ids, attention_mask = [b.to(device) for b in batch]
    model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    generated_ids = model.generate(**model_inputs, max_new_tokens=32)
    decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for j, item in enumerate(decoded_responses):
        # answer = item.split('\nmodel')[1]
        # question = item[5:].split('\nmodel')[0]
        print(json.dumps({'INDEX': i * len(decoded_responses) + j, 'OUTPUT': item}))

    # Free up memory
    del input_ids, attention_mask, generated_ids, decoded_responses
    torch.cuda.empty_cache()  # Use cautiously
    