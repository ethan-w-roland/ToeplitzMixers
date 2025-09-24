import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk
from linear_attention_transformer import LinearAttentionTransformerLM
from dotenv import load_dotenv
import safetensors
import pathlib
import torch.distributed._shard.checkpoint as dist_cp

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class UnrolledAutoencodingTransformer(nn.Module):

	def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512, compression=1, random=False, freeze_encoder=False, encoder_pos=None, decoder_pos=None):
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
		assert dim >= tokenized_length
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
		self.encoder_pos = encoder_pos
		self.decoder_pos = decoder_pos
	

	def forward(self, input_ids, labels=None, attention_mask=None):
		if self.random_input:
			x = torch.randint(1, self.n_vocab, input_ids.shape)
		else:
			x = input_ids
		x = x.to(device).squeeze(1)
		x = self.wte(x)
		if self.encoder_pos:
			x = x + self.encoder_pos(x).type(x.type()) # matches linear transformer default
		x = self.encoder(x, attention_mask=attention_mask, labels=labels)

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
		if self.decoder_pos:
			x = x + self.decoder_pos(x).type(x.type()) # matches linear transformer default
		x = self.decoder(x, attention_mask=attention_mask, labels=labels)

		output = self.lm_head(x)
		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output
	

class LinearTransformer(nn.Module):

        def __init__(self, vocab_size, dim, model):
                super().__init__()
                #self.lm_head = nn.Linear(dim, vocab_size, bias=False)
                self.longformer_model = model
                self.cel = nn.CrossEntropyLoss()

        def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, **kwargs):
                x = self.longformer_model(input_ids.to(device))
                output = x
                if labels.dim() > 2:
                        labels = rearrange(labels, 'b p t -> b (p t)')
                output = rearrange(output, 'b t e -> b e t')
                shift_logits = output[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous() 
                loss = self.cel(shift_logits, shift_labels)
                return loss, output

class TruncatedModel(nn.Module):
		def __init__(self, model, autoencoder=True):
				super().__init__()
				self.model_wte = model.longformer_model.token_emb
				self.model_blocks = model.longformer_model.transformer.layers

		def forward(self, x, **args):
				x = self.model_wte(x.to(device))
				for block in self.model_blocks:
						x = block(x)
				output = x 
				return output

class AbbreviatedModel(nn.Module):

        def __init__(self, model, depth=16, tokenized_length=512):
                super().__init__()
                if isinstance(model, LlamaForCausalLM):
                	self.model = model.model
                elif isinstance(model, LlamaModel):
                        self.model = model
                else:
                        raise TypeError('model type not recognized')

                self.depth = depth
                self.position_ids = torch.tensor([[i for i in range(tokenized_length)]]).to(device)

        def forward(self, input_ids: torch.Tensor, **attention_mask: torch.Tensor):
                # 'input_ids' is actually a float tensor, post-wte transformation
                x = input_ids.to(device)
                position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
                position_embeddings = self.model.rotary_emb(x, position_ids)

                for i in range(self.depth):
                        x = self.model.layers[i](x, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                return x

if __name__ == "__main__":
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')
	device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print("Vocab size: ", n_vocab)

	vocab_size = 8000
	dim = 512
	model= LinearAttentionTransformerLM(
	    num_tokens = 8000,
	    dim = 512,
	    heads = 4,
	    depth = 16,
	    max_seq_len = 512,
	    causal = True,                  # auto-regressive or not
	    ff_dropout = 0.,               # dropout for feedforward
	    attn_layer_dropout = 0.,       # dropout right after self-attention layer
	    attn_dropout = 0.,             # dropout post-attention
	    emb_dim = 512,                  # embedding factorization, to save on memory
	    dim_head = 128,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
	    blindspot_size = 1,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
	    n_local_attn_heads = 4,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
	    local_attn_window_size = 1,   # receptive field of the local attention
	    reversible = False,              # use reversible nets, from Reformer paper
	    ff_chunks = 1,                  # feedforward chunking, from Reformer paper
	    ff_glu = False,                  # use GLU variant for feedforward
	    attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
	    shift_tokens = False             # add single token shifting, for great improved convergence
	)

	encoder = LinearTransformer(vocab_size, dim, model)
	
	vocab_size = 8000
	tokenized_length = 512
	decoder_dim = 512
	n_layers = 8
	n_heads = 4
	model= LinearAttentionTransformerLM(
	    num_tokens = 8000,
	    dim = 512,
	    heads = 4,
	    depth = 16,
	    max_seq_len = 512,
	    causal = True,                  # auto-regressive or not
	    ff_dropout = 0.,               # dropout for feedforward
	    attn_layer_dropout = 0.,       # dropout right after self-attention layer
	    attn_dropout = 0.,             # dropout post-attention
	    emb_dim = 512,                  # embedding factorization, to save on memory
	    dim_head = 128,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
	    blindspot_size = 1,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
	    n_local_attn_heads = 4,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
	    local_attn_window_size = 1,   # receptive field of the local attention
	    reversible = False,              # use reversible nets, from Reformer paper
	    ff_chunks = 1,                  # feedforward chunking, from Reformer paper
	    ff_glu = False,                  # use GLU variant for feedforward
	    attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
	    shift_tokens = False             # add single token shifting, for great improved convergence
	)
	decoder = LinearTransformer(vocab_size, dim, model)
	encoder_model = encoder.longformer_model.transformer
	decoder_model = decoder.longformer_model.transformer
	encoder_pe = encoder.longformer_model.pos_emb
	model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=512, compression=1, random=False, freeze_encoder=False, encoder_pos=encoder_pe, decoder_pos=encoder_pe)
	print (model)
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
	output_dir = f'{checkpoint_root}/fineweb_autoencoding_lintransformer\
_{dim}\
_n{n_layers}\
_c{tokenized_length}_b{batch_size}x{n_devices}'
	
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=2,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
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

