"""
@author: ethan-w-roland
@date: 2025-07-20
@desc: Toeplitz Mixer Model
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("medium")


@dataclass
class Config:
    vocab_size: int = 4096  # tiny bc simple stories tokenizer
    block_size: int = 512
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    n_layer: int = 8


class Toeplitz(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.block_size))
        self.bias = nn.Parameter(torch.zeros(config.block_size))

    def vector_to_matrix(self, v: torch.Tensor, seq_len: int) -> torch.Tensor:
        T = min(seq_len, v.numel())
        M = v.new_zeros((T, T))
        i, j = torch.triu_indices(T, T, offset=0, device=v.device)
        M[i, j] = v[j - i]
        return M

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)  # (B, E, T)
        B, E, T = x.shape
        W = self.vector_to_matrix(self.weight, T).to(x.dtype)  # (T, T)
        out = (x.reshape(B * E, T) @ W).view(B, E, T)
        out = out + self.bias[:T].view(1, 1, T)
        return out.transpose(1, 2)  # (B, T, E)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.embed_dim)
        self.mixer = Toeplitz(config)
        self.norm2 = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToeplitzMixerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp_emb(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.out_emb(x)
        return x

    @torch.inference_mode()
    def generate(self, x: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        block_size = self.config.block_size

        with torch.inference_mode():
            for _ in range(max_tokens):
                x_cond = x[:, -block_size:]  # crop to context
                logits = self.forward(x_cond)  # (B, Tc, V)
                next_id = logits[:, -1, :].argmax(dim=-1)  # (B,)
                x = torch.cat([x, next_id.unsqueeze(1)], dim=1)

        return x
