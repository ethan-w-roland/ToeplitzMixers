import os
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow
import torch.nn as nn

from datasets import load_dataset, load_from_disk
import transformers
from transformers import MambaConfig, Mamba2Config, MambaForCausalLM, Mamba2ForCausalLM
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import datasets
import warnings
import shutil
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available else 'cpu'

class MambaCLM(nn.Module):
   
   def __init__(self, model, dim, vocab_size, copy=False):
       self.copy = copy
       self.model = model
       self.lm_head = nn.Linear(dim, vocab_size)
       self.vocab_size = vocab_size

   def forward(self, input_ids, labels=None, **kwargs):
        if self.copy:
            input_ids = copy_dataset(input_ids)
            if labels is not None:
                labels = copy_dataset(labels)
        labels = labels[:, 1:].contiguous()
        x = self.model(input_ids)
        logits = self.lm_head(x)
        logits = logits[:, :-1].contiguous()

        if labels is not None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits

def copy_dataset(input_ids):
    n_ctx = len(input_ids[0])
    for i, input in enumerate(input_ids):
        first_half = input[:n_ctx//2]
        copied_halves = torch.cat((first_half, first_half))
        input_ids[i] = copied_halves
    return input_ids

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_stack_8k")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
print (vocab_size)
dim = 512
context_length = 1024
n_layers = 16
state_size = 256
num_heads = 8
head_dim = 128

config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': num_heads,
    'vocab_size': vocab_size,
    'state_size': state_size,
    'hidden_dropout_prob': 0,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'chunk_size': context_length,
    'num_heads': num_heads,
    'head_dim': head_dim
}

config = Mamba2Config(**config_kwargs)

model = Mamba2ForCausalLM(config).backbone
model = MambaCLM(model, dim, vocab_size, copy=True)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

train_path = f"{data_root}/fineweb-tokenized-train-c1024-8k"
test_path = f"{data_root}/fineweb-tokenized-test-c1024-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
print (model)
total_length = 0
print (train_dataset[0])
# descriptive name for output
batch_size = 32
n_gpus = torch.cuda.device_count()
output_dir = f'{data_root}/stack_mamba_{dim}_s{state_size}_n{n_layers}_c{context_length}_b{batch_size}x{n_gpus}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=8000,
        learning_rate=2e-4,
        bf16=True,
        eval_strategy='steps',
        output_dir=output_dir,
        optim='adamw_torch',
        overwrite_output_dir=True,
        max_steps=200000,
        ddp_find_unused_parameters=True
)

trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

model.train()
trainer.train()
#trainer.train('/home/bbadger/Desktop/fineweb_mamba_256_s64_n16_c512_b16x4/checkpoint-16000')


