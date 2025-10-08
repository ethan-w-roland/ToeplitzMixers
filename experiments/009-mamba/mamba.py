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


warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available else 'cpu'


tokenizer = AutoTokenizer.from_pretrained(f"/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
print (vocab_size)
dim = 256 
context_length = 512 
n_layers = 16
state_size = 64
num_heads = 8
head_dim = 64

config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': 4,
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

model = Mamba2ForCausalLM(config)


train_path = f"/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = f"/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

print (train_dataset[0])
# descriptive name for output
batch_size = 16
output_dir = f'/home/bbadger/Desktop/fineweb_mamba_{dim}_s{state_size}_n{n_layers}_c{context_length}_b{batch_size}x4'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=4000,
        learning_rate=2e-4,
        fp16=True,
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
trainer.train('/home/bbadger/Desktop/fineweb_mamba_256_s64_n16_c512_b16x4/checkpoint-16000')


