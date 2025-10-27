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
from dotenv import load_dotenv
from torch import einsum

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available else 'cpu'

#define a MLP Mixer based causal-language-model using weight masking
class LinearAttentionToeplitz(nn.Module):
    def __init__(self, dim: int, seq_len, n_heads):
        
        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, seq_len))
        self.bias = nn.Parameter(torch.zeros(seq_len))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def vector_to_matrix(self, v):
        """
        Given a vector v of shape (m,), returns an (m x m) matrix M
        where M[i, j] = v[j - i] if j >= i, and 0 otherwise.
        
        For example, if v = [a, b, c, d] then M will be:
        
        [ a  b  c  d ]
        [ 0  a  b  c ]
        [ 0  0  a  b ]
        [ 0  0  0  a ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(torch.arange(m, device=v.device),
                                torch.arange(m, device=v.device), 
                                indexing='ij')
        # j - i gives the offset into v. When j < i, we want a 0.
        M = torch.where(j >= i, v[j - i], torch.zeros(m, m, device=v.device, dtype=v.dtype))
        return M

    def causal_linear_attn(self, q, k, v, W, eps=1e-3):
        bucket_size=1
        b, h, e, dtype = *q.shape, q.dtype
        q = q.softmax(dim=-1)
        k = torch.exp(k).type(dtype).clone()
        toep_weight = torch.unsqueeze(self.weight, dim=-1) # toeplitz (per-token) weight injection into KV comp
        v *= toep_weight

        q = q * e ** -0.5

        bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
        b_q, b_k, b_v = map(bucket_fn, (q, k, v))

        b_k_sum = b_k.sum(dim=-2)
        b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

        context = einsum('bhnd,bhne->bhde', b_k, b_v)
        context = context.cumsum(dim = -3).type(dtype)

        D_inv = 1. / einsum('bhd,bhnd->bhn', b_k_cumsum, b_q).clamp(min = eps)
        attn = einsum('bhnd,bhde,bhn->bhne', b_q, context, D_inv)
        attn = attn.reshape(*q.shape)
        attn = self.out_proj(attn)
        return attn

    def to_qkv(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return q, k, v

    def forward(self, x):
        """
        x shape: (batch, embed_dim, seq_len)
        """
        x = rearrange(x, 'b e t -> b t e')
        W = self.vector_to_matrix(self.weight)
        q, k, v = self.to_qkv(x)
        attention = self.causal_linear_attn(q, k, v, W)
        x = rearrange(x, 'b t e -> b e t')
        return attention


class ToeplitzHeads(nn.Module):

    def __init__(self, dim, seq_len, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.proj_head = nn.ModuleList(
            [nn.Linear(dim, hidden_dim)
            for i in range(n_heads)]
            ).to(device)

        self.out_proj = nn.Linear(dim, dim)

        self.mixer_heads = nn.ModuleList(
            [LinearAttentionToeplitz(dim//n_heads, seq_len, n_heads)
        for i in range(n_heads)]
        )

    def forward(self, x: torch.tensor):
        hidden_layer = []
        x = rearrange(x, 'b e t -> b t e')
        # pre-concatenated out projection
        for head in range(self.n_heads):
            projection = self.proj_head[head](x)
            projection = rearrange(projection, 'b t e -> b e t')
            conv_projection = self.mixer_heads[head](projection)
            rearranged_conv = rearrange(conv_projection, 'b e t -> b t e')
            hidden_layer.append(rearranged_conv)

        # concatenate multi-headed output: note that out projs are included in each head
        hidden_layer = torch.cat(hidden_layer, dim=1)
        return hidden_layer


class MixerBlock(nn.Module):
    
    def __init__(
        self,
        hidden_dim:int,
        seq_len:int,
        expansion_factor:int=4,
        dropout:float=0.,
        heads=None
        ):

        super(MixerBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor

        #channel-norm
        self.channel_norm = nn.LayerNorm(hidden_dim)

        #channel-mixing layer
        self.channel_mixing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        )

        #token-norm
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mixing_layer = ToeplitzHeads(hidden_dim, 
            seq_len, 
            hidden_dim//heads,
            heads)


    def forward(self, x):
        res = x
        x = self.channel_norm(x)
        x = self.channel_mixing_layer(x)
        x = x + res

        res = x
        x = self.token_norm(x)
        x = x.transpose(1, 2)
        x = self.token_mixing_layer(x)
        x = x.transpose(1, 2)
        x = x + res
        return x

class ToepLinTransformer(nn.Module):
    
    def __init__(
        self,
        vocab_size:int,
        hidden_dim:int,
        seq_len:int,
        num_blocks:int,
        heads=None,
        tie_io=False):

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks

        # Input Embedding
        self.input_layer = nn.Embedding(vocab_size, hidden_dim)

        # Mixer Blocks
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(hidden_dim, seq_len, heads=heads) for _ in range(num_blocks)]
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie input and output layer weights
        if tie_io:
            self.output_layer.weight = self.input_layer.weight

        # Initialize weights
        self._init_weights()

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, LinearAttentionToeplitz):
                # Kaiming He initialization for Swish activation
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None, **kwargs):
        labels = labels[:, 1:].contiguous()
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x)
        logits = self.output_layer(x)
        logits = logits[:, :-1].contiguous()

        if not labels is None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)
            
            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

dim = 512
context_length = 512
n_layers = 32
n_heads = 4
model = ToepLinTransformer(vocab_size, dim, context_length, n_layers, heads=n_heads)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_lintranstoep_512_h4_c512_b32x4'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
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
trainer.train()
