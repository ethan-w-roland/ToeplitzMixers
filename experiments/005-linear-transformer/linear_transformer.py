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
from dotenv import load_dotenv
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

load_dotenv()
data_root = os.getenv("DATA_ROOT")
checkpoint_root = os.getenv("CHECKPOINT_ROOT")
tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
print (vocab_size)
dim = 128
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
    dim = dim,
    heads = n_heads,
    depth = 16,
    max_seq_len = 512,
    causal = True,                  # auto-regressive or not
    ff_dropout = 0.,               # dropout for feedforward
    attn_layer_dropout = 0.,       # dropout right after self-attention layer
    attn_dropout = 0.,             # dropout post-attention
    emb_dim = dim,                  # embedding factorization, to save on memory
    dim_head = dim // n_heads,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
    blindspot_size = 1,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
    n_local_attn_heads = 0,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
    local_attn_window_size = 1,    # receptive field of the local attention
    reversible = False,              # use reversible nets, from Reformer paper
    ff_chunks = 1,                  # feedforward chunking, from Reformer paper
    ff_glu = False,                  # use GLU variant for feedforward
    attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
    shift_tokens = False,             # add single token shifting, for great improved convergence
    use_toeplitz=False,
    use_inverse=True
)
model = LinearTransformer(vocab_size, dim, model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"
print (model)
datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
#train_dataset = train_dataset.rename_column('input_ids', 'x')
#test_dataset = test_dataset.rename_column('input_ids', 'x')
print (train_dataset[0])
# descriptive name for output
output_dir = f'{checkpoint_root}/finemath_linear_transformer_inv_128_n8_c512_b64x2'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=64,
	per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	learning_rate=2e-4, 
	fp16=False,
        bf16=True, 
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000,
        ddp_find_unused_parameters=True,
#	logging_steps=500,
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
#trainer.train(output_dir + '/checkpoint-28000')
#trainer.train("/home/bbadger/Desktop/fineweb_linear_transformer_l64_512_c512/checkpoint-88000")
