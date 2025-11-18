import os
from prettytable import PrettyTable
import torch
import torch.nn as nn
from einops import rearrange
import transformers
import mlflow
from transformers import AutoTokenizer, LlamaConfig, LlamaModel
import datasets
from datasets import load_dataset, load_from_disk
from toep_mixer_multiheaded import MLPMixer

from dotenv import load_dotenv
import safetensors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

class MixerHead(nn.Module):

	def __init__(self, dim, length, hidden_dim, n_heads=4, kernel=1):
		super().__init__()
		self.n_heads = n_heads
		self.proj_head = nn.ModuleList(
			[nn.Linear(dim, hidden_dim)
			for i in range(n_heads)]
			).to(device)

		self.convs = nn.ModuleList(
			[nn.Conv1d(length, length, kernel, padding='same')
			for i in range(n_heads)]
			)
		self.out_proj = nn.Linear(dim, dim)
	

	def forward(self, x: torch.tensor):
		for i in range(len(self.convs)):
			masked_conv = torch.tril(rearrange(self.convs[i].weight, 'f d p -> p f d'))
			self.convs[i].weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		hidden_layer = []

		for head in range(self.n_heads):
			projection = self.proj_head[head](x)
			conv_projection = self.convs[head](projection)
			hidden_layer.append(conv_projection)

		# concatenate and project multi-headed output
		hidden_layer = torch.cat(hidden_layer, dim=2)
		hidden_layer = self.out_proj(hidden_layer)
		return hidden_layer

class MixerBlock(nn.Module):

	def __init__(self, dim, length, causal=True, n_heads=0, kernel=1):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.n_heads = n_heads
		self.multiheaded = False
		if n_heads and n_heads > 0:
			self.multiheaded = True
			self.conv = MixerHead(dim, length, dim//n_heads, n_heads=n_heads, kernel=kernel) # proj dim matches outer
		else:
			self.conv = nn.Conv1d(length, length, kernel, padding='same')
		self.causal = causal

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		if self.causal and not self.multiheaded:
			# for CLM training, apply lower triangular mask to convolution weights
			masked_conv = torch.tril(rearrange(self.conv.weight, 'f d p -> p f d'))
			self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual

		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x

class AutoencodingMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, compression=1, kernel=1, n_heads=0, unroll=True, random=False, frozen_encoder=None):
		super().__init__()
		self.n_vocab = n_vocab
		self.wte = nn.Embedding(n_vocab, dim)
		if frozen_encoder:
			# enforce no grad on encoder
			for _, param in frozen_encoder.named_parameters():
				param.requires_grad = False
			self.encoderblocks = frozen_encoder.model_blocks.to(device)
	
		self.decoderblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length, 
				causal = True,
				n_heads = n_heads,
				kernel = kernel
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(dim, dim//compression)
			self.up = nn.Linear(dim//compression, dim)
		self.unroll = unroll
		self.dim = dim
		self.projection = nn.Linear(dim//2, dim)
		self.random_input = random

	def forward(self, input_ids, labels=None, **kwargs):
		if self.random_input:
			x = torch.randint(1, self.n_vocab, input_ids.shape)
		else:
			x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.encoderblocks:
			x = block(x)
		
		encoder_embedding = x[:, -2, :].unsqueeze(1)
		
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)

		if self.unroll:
			embedding_stack = []
			# sliding window unroll over hidden dim
			for i in range(self.tokenized_length):
				i %= self.dim
				sliding_window = encoder_embedding[..., i:i+self.dim//2]
				if i+self.dim//2 > self.dim:
					residual = i+self.dim//2 - self.tokenized_length
					# loop around to first index
					sliding_window = torch.cat((sliding_window, encoder_embedding[..., :residual]), dim=2)
				embedding_stack.append(sliding_window)
			encoder_embedding = torch.cat(embedding_stack, dim=1)
			encoder_embedding = self.projection(encoder_embedding)

		else:
			# repeat embedding in token dim
			encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)

		x = encoder_embedding
		for block in self.decoderblocks:
			x = block(x)
		
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
			if self.double_tokens:
				labels = labels.reshape(labels.shape[0], labels.shape[1]//2, 2)

		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output


class TruncatedModel(nn.Module):
		def __init__(self, model, autoencoder=True):
				super().__init__()
				self.model_wte = model.input_layer
				self.model_blocks = model.mixer_blocks   

		def forward(self, x, **args):
				x = self.model_wte(x.to(device))
				for block in self.model_blocks:
						x = block(x)
				output = x 
				return output

if __name__ == '__main__':
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')
	device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print("Vocab size: ", n_vocab)

	tokenized_length = 512
	dim = 512
	layers = 16
	n_heads = 4

	# frozen encoder init and load
	encoder = MLPMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False
	)
	print (encoder)
	#safetensors.torch.load_model(encoder, f'{checkpoint_root}/fineweb_flat_h4_toep_512_n16_c512_b32x4/checkpoint-200000/model.safetensors')
	frozen_encoder = TruncatedModel(encoder, autoencoder=False)

	compression = 1
	kernel=1
	heads=4
	model = autoencoder = AutoencodingMixer(n_vocab,
		dim, 
		layers, 
		tokenized_length, 
		compression=compression,
		n_heads=heads, 
		kernel=kernel, 
		unroll=True, 
		random=False,
		frozen_encoder=frozen_encoder
	)

	train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

	datasets.config.IN_MEMORY_MAX_SIZE = 50e9
	train_dataset = load_from_disk(train_path, keep_in_memory=None)
	test_dataset = load_from_disk(test_path, keep_in_memory=None)
	print(len(train_dataset), len(test_dataset))
	mlflow.end_run()
	print("training begun")
	print(encoder)

	batch_size = 32
	n_devices = 4
	# get number of devices (assumes that all visible devices are used for training)
	if torch.cuda.is_available():
		n_devices = torch.cuda.device_count()

	# descriptive name for output
	output_dir = f'{checkpoint_root}/fineweb_untrained_information\
_{dim}\
_n{layers}\
_c{tokenized_length}_b{batch_size}x{n_devices}'
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
		save_safetensors=True,
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

