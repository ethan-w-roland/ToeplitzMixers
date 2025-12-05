import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_from_disk
import mlflow
import os
from dotenv import load_dotenv

from toep_mixer_multiheaded import MixerBlock as MixerBlock
from frozen_toep_mixer_multiheaded import MixerBlock as FrozenMixerBlock
from toep_mixer_multiconv import MixerBlock as MultiBLock

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

class AutoencodingMixer(nn.Module):

	def __init__(self, n_vocab, 
			dim, 
			depth, 
			length, 
			compression=1, 
			double_tokens=False, 
			kernel=1, 
			n_heads=0, 
			unroll=True, 
			random=False, 
			frozen_encoder=None, 
			clm_encoder=False,
			frozen_toeplitz=True
		):
		super().__init__()
		self.double_tokens = double_tokens
		self.n_vocab = n_vocab
		if double_tokens:
			self.wte = nn.Linear(n_vocab, dim)
		else:
			self.wte = nn.Embedding(n_vocab, dim)
		if frozen_encoder:
			# enforce no grad on encoder
			for _, param in frozen_encoder.named_parameters():
				param.requires_grad = False
			self.encoderblocks = frozen_encoder.model_blocks.to(device)
			#self.wte = frozen_encoder.model_wte.to(device)
		else:
			if frozen_toeplitz:
				self.encoderblocks = nn.ModuleList(
					[FrozenMixerBlock(
					hidden_dim = dim,
					seq_len = length,
					heads = n_heads,
					)
				for i in range(depth)]
				).to(device)
			else:
				self.encoderblocks = nn.ModuleList(
				[MixerBlock(
					hidden_dim = dim,
					seq_len = length,
					heads = n_heads,
					)
				for i in range(depth)]
				).to(device)
	

		if frozen_toeplitz:
			self.decoderblocks = nn.ModuleList(
					[FrozenMixerBlock(
					hidden_dim = dim,
					seq_len = length,
					heads = n_heads,
					)
				for i in range(depth)]
				).to(device)
		else:
			self.decoderblocks = nn.ModuleList(
				[MixerBlock(
					hidden_dim = dim,
					seq_len = length,
					heads = n_heads,
					)
				for i in range(depth)]
				).to(device)
			
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(dim, dim//compression)
			self.up = nn.Linear(dim//compression, dim)
		self.unroll = unroll
		self.dim = dim
		self.clm_encoder = clm_encoder
		self.projection = nn.Linear(dim//2, dim)
		self.random_input = random

	def forward(self, input_ids, labels=None, **kwargs):
		if self.random_input:
			x = torch.randint(1, self.n_vocab, input_ids.shape)
		else:
			x = input_ids
		x = x.to(device)
		if self.double_tokens:
			x_pairs = x.reshape(x.shape[0], x.shape[1]//2, 2)
			# implements a two hot tensor
			inputs = torch.nn.functional.one_hot(x_pairs[:, :, 0], self.n_vocab) + \
					 torch.nn.functional.one_hot(x_pairs[:, :, 1], self.n_vocab)

		x = self.wte(x)
		for block in self.encoderblocks:
			x = block(x)
		
		if self.clm_encoder:
			encoder_embedding = x[:, -2, :].unsqueeze(1)
		else:
			encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		
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
					residual = i+self.dim//2 - self.dim
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
		if labels is not None and labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
			if self.double_tokens:
				labels = labels.reshape(labels.shape[0], labels.shape[1]//2, 2)

		output = rearrange(output, 'b t e -> b e t')
		if labels is not None:
			loss = self.cel(output, labels)
		else:
			loss = 0
		return loss, output


if __name__ == "__main__":
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')

	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print("Vocab size: ", n_vocab)

	vocab_size = 8000
	dim = 256
	depth = 16
	length = 512
	compression=1     
	kernel=1
	heads=4
	model = AutoencodingMixer(vocab_size, dim, depth, length, n_heads=heads, kernel=kernel, compression=compression, frozen_toeplitz=False)
	train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

	datasets.config.IN_MEMORY_MAX_SIZE = 50e9
	train_dataset = load_from_disk(train_path, keep_in_memory=None)
	test_dataset = load_from_disk(test_path, keep_in_memory=None)
	print(len(train_dataset), len(test_dataset))
	mlflow.end_run()

	batch_size = 32
	n_devices = 4
	# get number of devices (assumes that all visible devices are used for training)
	if torch.cuda.is_available():
		n_devices = torch.cuda.device_count()

	# descriptive name for output
	output_dir = f'{checkpoint_root}/fineweb_autoencoding_toep_h4\
_{dim}\
_n{depth}\
_c{length}_b{batch_size}x{n_devices}'
	
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=2,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		warmup_steps=500,
		eval_steps=4000,
		save_steps=8000,
		learning_rate=2e-4,
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
