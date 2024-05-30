# =============================================================================
# INITIALIZATION
# =============================================================================
import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
from peft import LoraConfig, TaskType, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel

# Set the current working directory
os.chdir('/home/dan/DeepLearning/mini_temporal')

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")
    parser.add_argument('--train_data_file', type=str, required=True)
    parser.add_argument('--eval_data_file', type=str, required=True)
    parser.add_argument('--model_context', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    return parser.parse_args()

args = parse_args()

# =============================================================================
# DATA LOADING
# =============================================================================
def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file]

train_ds_packed = load_jsonl(args.train_data_file)
eval_ds_packed = load_jsonl(args.eval_data_file)

# =============================================================================
# DATA LOADER SETUP
# =============================================================================
train_dataloader = DataLoader(train_ds_packed, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
eval_dataloader = DataLoader(eval_ds_packed, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

# for batch in train_dataloader:
#     print(batch)  # See what the batch actually contains
#     break  # Stop after the first print to inspect the format

# =============================================================================
# MODEL SETUP
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained('google/gemma-2b-it', device_map='auto', torch_dtype=torch.float16)
model = DataParallel(model).to(device)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it')
scaler = GradScaler()

# PEFT Configuration
target_layers = ['encoder.layer.*.attention.self.query', 'encoder.layer.*.attention.self.key']

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_layers  # Specify the target layers for adaptation
)

# =============================================================================
# TRAINING UTILITIES SETUP
# =============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99), eps=1e-5)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

def loss_fn(outputs, labels):
    return torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

# =============================================================================
# TRAINING LOOP
# =============================================================================
model.train()
# Assuming you've already defined 'model', 'device', 'tokenizer', and 'train_dataloader'
def train(model, train_dataloader, device):
    model.train()  # Set the model to training mode
    for batch in train_dataloader:
        # Ensure batch['input_ids'] is a tensor and move it to the correct device
        input_ids = torch.tensor(batch['input_ids']).to(device)
        
        # Optional: If you have labels, similarly process them
        # labels = torch.tensor(batch['labels']).to(device)

        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(input_ids=input_ids)
        loss = outputs.loss  # Assuming model outputs include 'loss'

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()
        optimizer.zero_grad()  # Clear the gradients of all optimized variables

        # Optional: Log loss or other metrics here
        print(f"Loss: {loss.item()}")

# Run the training function
train(model, train_dataloader, device)

# =============================================================================
# VALIDATION
# =============================================================================
@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0
    for batch in eval_dataloader:
        inputs = tokenizer(batch['input_ids'], return_tensors='pt', padding=True, truncation=True).to(device)
        labels = tokenizer(batch['labels'], return_tensors='pt', padding=True, truncation=True).to(device)
        
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels['input_ids'])
        total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_dataloader)
    print(f"Validation Loss: {avg_loss}")
    model.train()

validate()

# =============================================================================
# MODEL SAVING
# =============================================================================
torch.save(model.state_dict(), f'./models/gemma_{args.model_context}_model.pth')
print(f"Model saved at './models/gemma_{args.model_context}_model.pth'")
