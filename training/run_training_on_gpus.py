# # """
# # Created on 04/09/2024

# # @author: Dan Schumacher
# # """
import torch
import os

# Explicitly set which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # This should make GPU 1 visible as the only GPU as 'cuda:0'

# Initialize device
torch.cuda.set_device(0)  # Since CUDA_VISIBLE_DEVICES is set to '1', 'cuda:0' should now refer to GPU 1
device = torch.device("cuda")  # Should now refer to GPU 1

print(f"Using device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # Should print the name of GPU 1


#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import os
os.chdir('/home/dan/mini_temporal')

import json
import torch


# Set this right after importing torch
# torch.cuda.set_device(1)  # Explicitly set to GPU 1

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from transformers import GenerationConfig
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
from peft import LoraConfig, TaskType, get_peft_model


#endregion
#region # COMMAND LINE ARGUMENTS
# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

    parser.add_argument('--train_data_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--eval_data_file', type=str, required=True, help='Path to evaluation data file')
    parser.add_argument('--model_context', type=str, required=True, help='Model context')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='How many pass at model at once?')
    parser.add_argument('--epochs', type=int, required=False, default=3, help= 'how many epochs do you wish to train for?')

    return parser.parse_args()

args = parse_args()
print('ARG PARSE DONE')

print()

# endregion
# region # DATA LOADING
# # =============================================================================
# # DATA LOADING
# # =============================================================================
def load_jsonl(filename):
    data = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filename}.")
        exit(1)
    return data

train_ds_packed = load_jsonl(args.train_data_file) 
train_ds_packed[:24]
eval_ds_packed = load_jsonl(args.eval_data_file) 
train_ds_packed[:24]

print('DATALOADED DONE')

#endregion
#region # DATA LOADER 
# =============================================================================
# DATA LOADER
# =============================================================================
batch_size = args.batch_size  

train_dataloader = DataLoader(
    train_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator,
    shuffle=True
)

eval_dataloader = DataLoader(
    eval_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator,
    shuffle=False,
)

#  CHECK TO SEE WHAT THE BATCH LOOKS LIKE
b = next(iter(train_dataloader))
# b.keys(), b['input_ids'][0][:25], b['labels'][0][:25]
# b['input_ids'].shape
# b['labels'].shape
print('DATALOADERS DONE')

#endregion
#region # MODEL CONFIG
# =============================================================================
# MODEL CONFIG
# =============================================================================

gradient_accumulation_steps = 32 // batch_size

# Define base model paths
model_id = 'google/gemma-2b-it'

max_seq_len = 1024

# Use the base paths to set the correct model_id
config = SimpleNamespace(
    model_id=model_id,
    dataset_name=f"gemma_{args.model_context}_context",
    precision="bf16",                                         # faster and better than fp16, requires new GPUs
    n_freeze=24,                                              # How many layers we don't train, LLama 7B has 32.
    lr=2e-4,                                                  # the learning rate
    n_eval_samples=10,                                        # How many samples to generate on validation
    max_seq_len=max_seq_len,                                  # Length of the sequences to pack
    epochs=args.epochs,                                       # we do 3 pasess over the dataset.
    gradient_accumulation_steps=gradient_accumulation_steps,  # evey how many iterations we update the gradients, simulates larger batch sizes
    batch_size=batch_size,                                    # what my GPU can handle, depends on how many layers are we training  
    log_model=False,                                          # upload the model to W&B?
    mom=0.9,                                                  # optim param
    gradient_checkpointing = True,                            # saves even more memory
    freeze_embed = True,                                      # why train this? let's keep them frozen ❄️
)

config.total_train_steps = config.epochs * len(train_dataloader) // config.gradient_accumulation_steps

with open('./training/token.txt', 'r') as file:
    # Read the entire content of the file into a single string
    token = file.read()

# model = AutoModelForCausalLM.from_pretrained(
#     config.model_id,
#     device_map='cuda', # Which GPU? 0 or 1?
#     trust_remote_code=True,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.bfloat16,
#     use_cache=False,
#     token = token,
#     # attn_implementation="flash_attention_2"
# )


model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    device_map='auto',  # Automatically distribute the model to the available GPUs
    torch_dtype=torch.bfloat16,  # Use bfloat16 for computation
    use_cache=False
)


print('MODEL CONFIG DONE')
#endregion
#region # PEFT
# =============================================================================
# PEFT
# =============================================================================
peft_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    inference_mode = False,
    r = 8,
    lora_alpha = 32,
    lora_dropout = 0.1
)
model = get_peft_model(model, peft_config)

# model.print_trainable_parameters()
print('PEFT DONE')

#endregion
#region # FREEZING
# =============================================================================
# FREEZING
# =============================================================================

# # Freeze all parameters initially
# for param in model.parameters(): 
#     param.requires_grad = False

# # Unfreeze lm_head parameters
# for param in model.base_model.model.lm_head.parameters(): 
#     param.requires_grad = True

# # Unfreeze certain layers
# n_freeze = 12  # Adjust the number of layers you want to freeze
# for i, layer in enumerate(model.base_model.model.model.layers):
#     if i >= n_freeze:
#         for param in layer.parameters():
#             param.requires_grad = True

# # Freeze the Embeddings
# if config.freeze_embed:
#     model.base_model.model.model.embed_tokens.weight.requires_grad_(False)

# # Enable Gradient Checkpointing
# if config.gradient_checkpointing:
#     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print('FREEZING DONE')

#endregion
#region # OPTIMIZER AND SCHEDULER
# =============================================================================
# OPTIMIZER AND SCHEDULER
# =============================================================================
optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9,0.99), eps=1e-5)
scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_training_steps=config.total_train_steps,
    num_warmup_steps=config.total_train_steps // 10,
)

def loss_fn(x, y):
    "A Flat CrossEntropy" 
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))
print('OPTIMSCHEDULER DONE')

#endregion
#region # SAMPLING FROM THE MODEL
# =============================================================================
# SAMPLING FROM THE MODEL
# =============================================================================
gen_config = GenerationConfig.from_pretrained(config.model_id)

# create simple sample function
# see what the model is outputting
config.model_id
tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=token) # NEEDS TO BE COMMANDLINE ARG / & only use token if gemma
tokenizer.pad_token = tokenizer.eos_token	

def generate(prompt, max_new_tokens=100, gen_config=gen_config):
    with torch.inference_mode():
        tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        output = model.generate(tokenized_prompt, 
                            max_new_tokens=max_new_tokens, 
                            generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

# Define a configuration for testing
test_config = SimpleNamespace(
    max_new_tokens=50,  # Adjust according to your model's capabilities
    gen_config=SimpleNamespace(temperature=0.9, top_p=0.95)  # Example values
)
print('SAMPLING DONE')

#endregion
#region # VALIDATION STEP
# =============================================================================
# VALIDATION STEP
# =============================================================================
@torch.no_grad()
def validate():
    model.eval()
    eval_loss = 0
    # eval_acc = Accuracy()  # Uncomment and ensure Accuracy is a suitable metric class if needed

    for batch in eval_dataloader:
        batch = to_gpu(batch)
        out = model(**batch)
        loss = loss_fn(out.logits, batch["labels"])
        # eval_acc.update(out.logits, batch["labels"])
        eval_loss += loss.item()

    avg_loss = eval_loss / len(eval_dataloader)
    # avg_acc = eval_acc.compute()

    print(f"Validation Loss: {avg_loss:.4f}")
    # print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_acc:.4f}")

    model.train()

print('VAL DONE')

#endregion
#region # TRAINING LOOP
# =============================================================================
# TRAINING LOOP
# =============================================================================
# Training

# def to_gpu(batch):
#     # Assuming your batch is a dictionary of tensors
#     thingy = {k: v.to('cuda') for k, v in batch.items()}
#     return thingy


def to_gpu(batch):
    # Assuming your batch is a dictionary of tensors
    return {k: v.to(device) for k, v in batch.items()}


# set model to training mode
model.train()

train_step = 0
pbar = tqdm(total=config.total_train_steps)

for epoch in range(config.epochs):
    # break # @$@
    for step, batch in enumerate(train_dataloader):
        batch = to_gpu(batch)  # Ensure your batch is properly moved to GPU
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps
            loss.backward()
        if step % config.gradient_accumulation_steps == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            # Logging the metrics without W&B
            current_loss = loss.item() * config.gradient_accumulation_steps
            # current_acc = acc.update(out.logits, batch["labels"])
            current_lr = scheduler.get_last_lr()[0]

            print(f"Epoch: {epoch}, Step: {train_step}, Loss: {current_loss:.4f}")
            train_step += 1
            pbar.update(1)

    # Perform validation and print validation metrics
    validate()

pbar.close()

print('TRAINING DONE')


#endregion
#region # SAVING THE MODEL
# =============================================================================
# SAVING THE MODEL
# =============================================================================

# Model name (gemma or llama)

file_path = f'./training/models/mini_{args.model_context}_context_model.pt'

torch.save(model, file_path)
print(f'model saved at {file_path}')
torch.cuda.empty_cache()

state_dict_path = file_path.replace('.pt', '_state_dict.pt')  # Assuming file_path ends with '.pt'
torch.save(model.state_dict(), state_dict_path)
print(f"State dictionary saved at {state_dict_path}")

#endregion