import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, Mamba2Config, Mamba2Model 
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
import safetensors
import pathlib
import torch.distributed._shard.checkpoint as dist_cp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class UnrolledAutoencodingMamba(nn.Module):

	def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512, compression=1, random=False, freeze_encoder=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.encoder = encoder_model
		if freeze_encoder:
			for _, param in encoder_model.named_parameters():
				param.requires_grad = False

		self.decoder = decoder_model
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = tokenized_length
		unroll_factor = dim // tokenized_length #assumes
		self.projection = nn.Linear(dim//2, dim)
		self.dim = dim
		self.compression=False
		if compression > 1:
			self.down = nn.Linear(dim, dim//compression)
			self.up = nn.Linear(dim//compression, dim)
			self.compression=True
		self.random_input = random
		self.n_vocab = n_vocab
	

	def forward(self, input_ids, labels=None, attention_mask=None):
		if self.random_input:
			x = torch.randint(1, self.n_vocab, input_ids.shape)
		else:
			x = input_ids
		x = x.to(device).squeeze(1)
		x = self.encoder(x, attention_mask=attention_mask, labels=labels).last_hidden_state

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)
		embedding_stack = []
		# sliding window unroll over hidden dim
		for i in range(self.tokenized_length):
			sliding_window = encoder_embedding[..., i:i+self.dim//2]
			if i+self.dim//2 > self.dim:
				residual = i+self.dim//2 - self.tokenized_length
				# loop around to first index
				sliding_window = torch.cat((sliding_window, encoder_embedding[..., :residual]), dim=2)
			embedding_stack.append(sliding_window)
		encoder_embedding = torch.cat(embedding_stack, dim=1)
		encoder_embedding = self.projection(encoder_embedding)
		x = encoder_embedding
		x = self.decoder(x, attention_mask=attention_mask, labels=labels)

		output = self.lm_head(x)
		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output
	
load_dotenv()
checkpoint_root = '/home/bbadger/Desktop'
data_root = '/home/bbadger/Desktop'
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print("Vocab size: ", n_vocab)

vocab_size=8000
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

encoder_model = Mamba2Model(config)
decoder_model = Mamba2Model(config)

model = UnrolledAutoencodingMamba(vocab_size, dim, encoder_model, decoder_model, tokenized_length=512, compression=1, random=False, freeze_encoder=False)
print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
print(len(train_dataset), len(test_dataset))
mlflow.end_run()

batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_autoencoding_mamba2\
_{dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

training_arguments = transformers.TrainingArguments(
	num_train_epochs=2,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	learning_rate=5e-4,
	fp16=True,
	eval_strategy="steps",
	output_dir=output_dir,
	optim="adamw_torch",
	overwrite_output_dir=True,
	save_safetensors=False,
	max_steps=200000,
)

trainer = transformers.Trainer(
	model=model.to("cuda"),  # pre-assignment for FSDP initialization
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train()


