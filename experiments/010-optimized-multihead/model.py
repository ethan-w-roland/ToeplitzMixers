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
import shutil
from dataclasses import dataclass
from typing import Tuple
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int
    embed_dim: int
    seq_len: int
    num_blocks: int
    n_heads: int
    mlp_dim: int
    dropout: float


def vector_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """
    Given a batched vector v of shape (n_heads, m), returns an (n_heads, m, m) tensor
    where output[h, i, j] = v[h, j - i] if j >= i, and 0 otherwise.
    
    This creates a Toeplitz causal matrix for each head in parallel.
    """
    n_heads, m = v.shape
    # Create index grids for rows and columns
    i, j = torch.meshgrid(
        torch.arange(m, device=v.device),
        torch.arange(m, device=v.device),
        indexing="ij",
    )
    
    offset = j - i  # shape: (m, m)
    
    M = torch.where(
        j >= i,  # shape: (m, m), broadcasted to (n_heads, m, m)
        v[:, offset],  # shape: (n_heads, m, m)
        torch.zeros(n_heads, m, m, device=v.device, dtype=v.dtype)
    )
    return M


class MultiHeadMixer(nn.Module):

    def __init__(
        self,
        config: Config
    ):
        super().__init__()

        self.n_heads = config.n_heads
        self.seq_len = config.seq_len
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.embed_dim // config.n_heads
        
        # Toeplitz parameters for all heads
        self.weight = nn.Parameter(torch.randn(config.n_heads, config.seq_len))
        self.bias = nn.Parameter(torch.zeros(config.n_heads, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, embed_dim, seq_len)
        """
        B, E, S = x.shape
        H = self.n_heads
        D = self.hidden_dim
        
        # Split into heads: (B, E, S) -> (B, H, D, S)
        x = x.view(B, H, D, S)
        
        # Create Toeplitz matrices for all heads: (H, S, S)
        W = vector_to_matrix(self.weight)
        
        # Apply Toeplitz mixing: (B, H, D, S) @ (H, S, S) -> (B, H, D, S)
        # Flatten batch and heads for bmm: (B*H, D, S) @ (B*H, S, S) -> (B*H, D, S)
        x = x.reshape(B * H, D, S)
        W = W.repeat_interleave(B, dim=0)  # (H, S, S) -> (B*H, S, S)
        x = torch.bmm(x, W)
        
        # Add bias: (H, S) broadcasted to (B*H, D, S)
        x = x + self.bias.repeat_interleave(B, dim=0).unsqueeze(1)
        
        # Reshape back to (B, E, S)
        x = x.reshape(B, E, S)
        
        return x


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):

    def __init__(self, config: Config):

        super().__init__()

        self.mlp_norm = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)
        self.mixer_norm = nn.RMSNorm(config.embed_dim)
        self.mixer = MultiHeadMixer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.mlp_norm(x))
        x = x + self.mixer(self.mixer_norm(x))
        return x

class MLPMixer(nn.Module):

    def __init__(self, config: Config):

        super().__init__()

        # Input Embedding
        self.inp_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)])
        self.out_embed = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, MultiHeadMixer):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:

        x = self.inp_embed(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.out_embed(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )

        return logits, loss

# device = "cuda" if torch.cuda.is_available() else "cpu"

# if __name__ == "__main__":
#     load_dotenv()
#     checkpoint_root = os.getenv('CHECKPOINT_ROOT')
#     data_root = os.getenv('DATA_ROOT')
#     tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
#     tokenizer.pad_token = tokenizer.eos_token
#     n_vocab = len(tokenizer)
#     print("Vocab size: ", n_vocab)

#     tokenized_length = 1024
#     dim = 1024
#     layers = 16
#     n_heads = 4

#     model = MLPMixer(
#         n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False
#     ).float()

#     train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
#     test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"

#     output_dir = f"{checkpoint_root}/fineweb_flat_h4_toep_1024_n16_c1024_b16x4"
    
#     datasets.config.IN_MEMORY_MAX_SIZE = 50e9
#     train_dataset = load_from_disk(train_path, keep_in_memory=None)
#     test_dataset = load_from_disk(test_path, keep_in_memory=None)
#     print(len(train_dataset), len(test_dataset))
#     mlflow.end_run()
#     print("training begun")
#     print(model)
#     training_arguments = transformers.TrainingArguments(
#         num_train_epochs=2,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         warmup_steps=500,
#         eval_steps=4000,
#         save_steps=8000,
#         learning_rate=5e-4,
#         fp16=True,
#         eval_strategy="steps",
#         output_dir=output_dir,
#         optim="adamw_torch",
#         overwrite_output_dir=True,
#         save_safetensors=True,
#         max_steps=200000,
#     )

#     trainer = transformers.Trainer(
#         model=model.to("cuda"),  # pre-assignment for FSDP initialization
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         args=training_arguments,
#         data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     )

#     # save driver code snapshot in checkpoint dir 
#     code_path = os.path.abspath(__file__) 
#     if not os.path.isdir(output_dir): 
#         os.mkdir(output_dir) 
#     shutil.copy(code_path, output_dir) 

#     model.train()
#     trainer.train()
