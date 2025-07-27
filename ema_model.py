"""
@author: ethan-w-roland
@date: 2025-07-27
@desc A small side experiment exploring EMA as a replacement for the mixer layer.
"""

from dataclasses import dataclass
import torch, torch.nn as nn

torch.set_float32_matmul_precision("medium")


# ----------------------------- Config ----------------------------- #

@dataclass
class Config:
    vocab_size: int = 4096          # tiny bc simple stories tokenizer
    block_size: int = 512           # context window (not required by EMA, but used in generate cropping)
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    n_layer: int = 8
    num_emas: int = 4               # number of parallel shared EMAs per block


# ----------------------- Mixture-of-EMAs Mixer -------------------- #

class EMAMixer(nn.Module):
    """
    Mixture of M shared EMAs.

    For each EMA m in {1..M} with scalar parameters (alpha_m, bias_m):
        y_t^{(m)} = alpha_m * y_{t-1}^{(m)} + (1 - alpha_m) * x_t + bias_m

    Full-sequence (training) path computes all M streams in parallel:
        - y^{(m)} ∈ R^{B,T,E} for all m  -> concatenate along E: [B,T,E*M]
        - project with Linear(E*M -> E)

    Decode path keeps cached states: state ∈ R^{B, M, E}.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_emas = config.num_emas

        # Alpha parameters per EMA (scalars), parameterized via sigmoid
        init_alpha = 0.9
        inv_sig = torch.log(torch.tensor(init_alpha) / (1 - torch.tensor(init_alpha))) # inverse sigmoid
        self.alpha_logits = nn.Parameter(inv_sig.expand(self.num_emas).clone()) # (M,)

        # Bias per EMA (scalar)
        self.bias = nn.Parameter(torch.zeros(self.num_emas)) # (M,)

        # Down-projection after concatenation of M streams
        self.proj = nn.Linear(self.embed_dim * self.num_emas, self.embed_dim, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, E = x.shape
        M = self.num_emas
        a = torch.sigmoid(self.alpha_logits) # (M,)
        s = (1.0 - a) # (M,)

        # Work in (B, E, T) then vectorize over M
        x = x.transpose(1, 2) # (B, E, T)
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0) # [0, 1, ..., T-1]

        # (M, T)
        loga = torch.log(a).unsqueeze(1)
        a_neg_pow = torch.exp(-loga * t) # alpha^{-k}
        a_pos_pow = torch.exp( loga * t) # alpha^{t}

        # Weighted cumsum per EMA:
        # z_t^{(m)} = sum_{k<=t} (1-a_m) * a_m^{-k} * x_k
        # Broadcast to (B, M, E, T)
        weighted = x.unsqueeze(1) * s.view(1, M, 1, 1) * a_neg_pow.view(1, M, 1, T)
        z = torch.cumsum(weighted, dim=-1) # (B, M, E, T)

        y = z * a_pos_pow.view(1, M, 1, T) # (B, M, E, T)

        # Bias sequence per EMA:
        # b * (1 - a^{t+1}) / (1 - a)
        a_t1 = a_pos_pow * a.view(M, 1) # (M, T) -> a^{t+1}
        denom = s.clamp_min(1e-8) # (M,)
        bias_seq = self.bias.view(M, 1) * (1.0 - a_t1) / denom.view(M, 1) # (M, T)
        y = y + bias_seq.view(1, M, 1, T)

        # y: (B, M, E, T) -> (B, T, E*M)
        y = y.permute(0, 3, 2, 1).contiguous().view(B, T, E * M)

        # Down-project
        y_proj = self.proj(y)                          # (B, T, E)
        return y_proj


    def step(self, x_t: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One-step EMA update for all M filters.

        Args:
            x_t:   (B, E)   current input
            state: (B, M, E) previous EMA outputs per filter; None -> zeros

        Returns:
            y_proj_t: (B, E)     projected mixed output for this step
            new_state: (B, M, E) updated states per EMA
        """
        B, E = x_t.shape
        M = self.num_emas
        if state is None:
            state = x_t.new_zeros(B, M, E)

        a = torch.sigmoid(self.alpha_logits).view(1, M, 1) # (1, M, 1)
        s = (1.0 - a) # (1, M, 1)
        b = self.bias.view(1, M, 1) # (1, M, 1)

        x_exp = x_t.view(B, 1, E) # (B, 1, E)
        y_t = a * state + s * x_exp + b # (B, M, E)

        y_cat = y_t.permute(0, 2, 1).contiguous().view(B, E * M) # (B, E*M)
        y_proj_t = self.proj(y_cat) # (B, E)

        return y_proj_t, y_t


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
        self.mixer = EMAMixer(config)
        self.norm2 = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, h_t: torch.Tensor, state: torch.Tensor | None):
        """
        Single-token update.

        Args:
            h_t:   (B, E)        current hidden into this block
            state: (B, M, E) or None   cached EMA states for this block

        Returns:
            h_next: (B, E)
            new_state: (B, M, E)
        """
        n1_t = self.norm1(h_t)
        mix_t, new_state = self.mixer.step(n1_t, state)  # (B,E), (B,M,E)
        h_t = h_t + mix_t
        n2_t = self.norm2(h_t)
        h_t = h_t + self.mlp(n2_t)
        return h_t, new_state


# ----------------------------- Model ----------------------------- #

class EMAMixerModel(nn.Module):
    """
    Same external name for drop-in use. Internally uses Mixture-of-EMAs mixer.
    """
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
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

    # -------- Full forward (training / eval) -------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) token ids
        """
        x = self.inp_emb(x)                      # (B,T,E)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.out_emb(x)                      # (B,T,V)
        return x

    # -------- Prefill to build EMA states -------- #
    def _prefill(self, x: torch.Tensor):
        """
        Process prompt and return:
          - logits over prompt, (B,T,V)
          - list of per-block EMA states, each (B, M, E)
          - last hidden vector after final block, (B,E)
        """
        emb = self.inp_emb(x) # (B,T,E)
        h = emb
        states = []
        for blk in self.blocks:
            n1 = blk.norm1(h) # (B,T,E)
            mix = blk.mixer(n1) # (B,T,E)

            n1_last = n1[:, -1, :] # (B,E)
            _, block_state = blk.mixer.step(n1_last, None) # (B,M,E)
            states.append(block_state)

            h = h + mix
            n2 = blk.norm2(h)
            h = h + blk.mlp(n2)

        h_last = h[:, -1, :] # (B,E)
        h = self.norm(h)
        logits = self.out_emb(h) # (B,T,V)
        return logits, states, h_last

    # -------- Greedy generation with O(1) updates -------- #
    @torch.inference_mode()
    def generate(self, x: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """
        Greedy decode.

        Args:
            x: (B,T) or (T,)
        Returns:
            tokens: (B, T + max_tokens)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)

        # For consistency with other models, crop to context window if needed
        if x.size(1) > self.config.block_size:
            x = x[:, -self.config.block_size:]

        logits, states, _ = self._prefill(x)
        B = x.size(0)

        for _ in range(max_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1)     # (B,)
            x = torch.cat([x, next_id.unsqueeze(1)], dim=1)

            # One-step through blocks with cached states
            h_t = self.inp_emb(next_id)                   # (B,E)
            new_states = []
            for blk, state in zip(self.blocks, states):
                h_t, state = blk.step(h_t, state)         # (B,E), (B,M,E)
                new_states.append(state)
            states = new_states

            h_t = self.norm(h_t)
            logits_t = self.out_emb(h_t).unsqueeze(1)     # (B,1,V)
            logits = torch.cat([logits, logits_t], dim=1)

        return x