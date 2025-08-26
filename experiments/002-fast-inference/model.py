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

    @torch.inference_mode()
    def step(
        self,
        n1_t: torch.Tensor,  # (B, E) normalized new feature
        hist: torch.Tensor | None,  # (B, E, Tprev) or None
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if hist is None:
            hist = n1_t.unsqueeze(-1)  # (B,E,1)
        else:
            hist = torch.cat([hist, n1_t.unsqueeze(-1)], dim=-1)  # (B,E,Tprev+1)

        if hist.size(-1) > max_len:
            hist = hist[:, :, -max_len:].contiguous()

        T = hist.size(-1)
        w = self.weight[:T]  # (T,)
        y_t = torch.einsum("bet,t->be", hist, torch.flip(w, dims=[0]))
        y_t = y_t + self.bias[T - 1]  # scalar bias for current time
        return y_t, hist


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
    def generate_old(self, x: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """
        Un-cached greedy autoregressive generation
        """

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

    @torch.inference_mode()
    def generate(self, x: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """
        Cached greedy autoregressive generation
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        block_size = self.config.block_size

        # ---- PREFILL ----
        x = x[:, -block_size:]

        # run one forward pass, building per-block histories of norm1(x)
        caches = []
        h = self.inp_emb(x)  # (B,T,E)
        for blk in self.blocks:
            n1 = blk.norm1(h)  # (B,T,E)
            hist = n1.transpose(1, 2).contiguous()  # (B,E,T)
            caches.append(hist)
            mix_full = blk.mixer(n1)  # (B,T,E)
            h = h + mix_full
            n2 = blk.norm2(h)
            h = h + blk.mlp(n2)

        h = self.norm(h)
        logits = self.out_emb(h)  # (B,T,V)

        # ---- DECODE ----
        for _ in range(max_tokens):
            next_ids = logits[:, -1, :].argmax(dim=-1)  # (B,)
            x = torch.cat([x, next_ids.unsqueeze(1)], dim=1)

            # single-step through blocks using step()
            h_t = self.inp_emb(next_ids)  # (B,E)
            new_caches = []
            for blk, cache in zip(self.blocks, caches):
                n1_t = blk.norm1(h_t)  # (B,E)
                y_t, new_hist = blk.mixer.step(
                    n1_t, cache, block_size
                )  # (B,E), (B,E,T)
                h_t = h_t + y_t
                n2_t = blk.norm2(h_t)
                h_t = h_t + blk.mlp(n2_t)
                new_caches.append(new_hist)
            caches = new_caches

            h_t = self.norm(h_t)
            logits_t = self.out_emb(h_t).unsqueeze(1)  # (B,1,V)
            logits = torch.cat([logits, logits_t], dim=1)

        return x
