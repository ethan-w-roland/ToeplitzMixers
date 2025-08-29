import os
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow
import torch.nn as nn

from datasets import load_dataset, load_from_disk
import transformers
from transformers import LongformerConfig, LongformerModel
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import datasets
import warnings
import shutil
from transformers.models.longformer import LongformerModel
from linear_attention_transformer import LinearAttentionTransformerLM

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available else 'cpu'


class LinearTransformerModel(nn.Module):

	def __init__(self, vocab_size, dim, longformer_model):
		super().__init__()
		self.lm_head = nn.Linear(dim, vocab_size, bias=False)
		self.longformer_model = longformer_model

	def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, **kwargs):
		x = self.longformer_model(input_ids.to(device), attention_mask=attention_mask)
		output = self.lm_head(x.last_hidden_state)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous() 
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

class LinearTransformer(nn.Module):

        def __init__(self, vocab_size, dim, longformer_model):
                super().__init__()
                self.lm_head = nn.Linear(dim, vocab_size, bias=False)
                self.longformer_model = longformer_model
                self.cel = nn.CrossEntropyLoss()

        def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, **kwargs):
                x = self.longformer_model(input_ids.to(device))
                output = x#self.lm_head(x)
                if labels.dim() > 2:
                        labels = rearrange(labels, 'b p t -> b (p t)')
                output = rearrange(output, 'b t e -> b e t')
                shift_logits = output[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous() 
                loss = self.cel(shift_logits, shift_labels)
                return loss, output


tokenizer = AutoTokenizer.from_pretrained(f"/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
print (vocab_size)
dim = 512
context_length = 512
n_layers = 16
n_heads = 4

config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': 4,
    'vocab_size': vocab_size,
    'hidden_dropout_prob': 0,
    'attention_window': 128
}

#configuration = LongformerConfig(**config_kwargs)
#model = LinearTransformerModel(vocab_size, dim, LongformerModel(configuration))
#print (model)
model = LinearAttentionTransformerLM(
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
    blindspot_size = 32,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
    n_local_attn_heads = 4,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
    local_attn_window_size = 64,   # receptive field of the local attention
    reversible = False,              # use reversible nets, from Reformer paper
    ff_chunks = 1,                  # feedforward chunking, from Reformer paper
    ff_glu = False,                  # use GLU variant for feedforward
    attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
    shift_tokens = True             # add single token shifting, for great improved convergence
)
model = LinearTransformer(vocab_size, dim, model)
train_path = f"/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = f"/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
#train_dataset = train_dataset.rename_column('input_ids', 'x')
#test_dataset = test_dataset.rename_column('input_ids', 'x')
print (train_dataset[0])
# descriptive name for output
output_dir = '/home/bbadger/Desktop/fineweb_linear_transformer_l64_512_c512_b32x4'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
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
#code_path = os.path.abspath(__file__)
#if not os.path.isdir(output_dir):
#	os.mkdir(output_dir)
#shutil.copy(code_path, output_dir)

model.train()
trainer.train("/home/bbadger/Desktop/fineweb_linear_transformer_l64_512_c512/checkpoint-88000")
