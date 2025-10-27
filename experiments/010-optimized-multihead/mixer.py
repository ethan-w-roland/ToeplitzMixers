import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int
    embed_dim: int
    seq_len: int
    num_blocks: int
    num_heads: int
    mlp_dim: int
    dropout: float
    do_toep_mean: bool
    do_toep_proj: bool


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

        self.n_heads = config.num_heads  # Fixed: use num_heads from config
        self.seq_len = config.seq_len
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.embed_dim // config.num_heads
        self.do_toep_mean = config.do_toep_mean
        self.do_toep_proj = config.do_toep_proj
        
        # Toeplitz parameters for all heads
        self.weight = nn.Parameter(torch.randn(config.num_heads, config.seq_len))
        self.bias = nn.Parameter(torch.zeros(config.num_heads, config.seq_len))
        
        if self.do_toep_proj:
            self.inp_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
            self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        parallel: bool
        ) -> torch.Tensor:
        """
        x shape: (batch, seq_len, embed_dim) from Block
        """
        B, S, E = x.shape
        H = self.n_heads
        D = self.hidden_dim
        
        # Apply input projection: (B, S, E) -> (B, S, E)
        if self.do_toep_proj:
            x = self.inp_proj(x)
        
        # Transpose to (B, E, S) for Toeplitz processing
        x = x.transpose(1, 2)
        
        # Split into heads: (B, E, S) -> (B, H, D, S)
        x = x.view(B, H, D, S)

        if parallel:
            
            # Create Toeplitz matrices for all heads: (H, S, S)
            # Use only the first S weights to match actual sequence length
            W = vector_to_matrix(self.weight[:, :S])
            
            # Apply Toeplitz mixing: (B, H, D, S) @ (H, S, S) -> (B, H, D, S)
            # Flatten batch and heads for bmm: (B*H, D, S) @ (B*H, S, S) -> (B*H, D, S)
            x = x.reshape(B * H, D, S)
            W = W.repeat_interleave(B, dim=0)  # (H, S, S) -> (B*H, S, S)
            x = torch.bmm(x, W)
            
            # Normalize by sum of weights to frame as weighted average
            if self.do_toep_mean:
                norm_factors = torch.cumsum(self.weight[:, :S], dim=-1)  # (H, S)
                norm_factors = norm_factors.repeat_interleave(B, dim=0).unsqueeze(1)  # (B*H, 1, S)
                x = x / norm_factors  # Broadcast: (B*H, D, S) / (B*H, 1, S)
            
            # Add bias: (H, S) broadcasted to (B*H, D, S)
            x = x + self.bias[:, :S].repeat_interleave(B, dim=0).unsqueeze(1)

        else:
            
            # Process each head separately
            for h in range(H):

                # Get Toeplitz matrix for this head: (S, S)
                # Use only the first S weights to match actual sequence length
                W_h = vector_to_matrix(self.weight[h:h+1, :S]).squeeze(0)
                
                # Apply to this head: (B, D, S) @ (S, S) -> (B, D, S)
                x_h = x[:, h, :, :]  # (B, D, S)
                x_h = torch.matmul(x_h, W_h)
                
                # Normalize by sum of weights to frame as weighted average
                if self.do_toep_mean:
                    norm_factors = torch.cumsum(self.weight[h, :S], dim=-1)  # (S,)
                    x_h = x_h / norm_factors  # Broadcast: (B, D, S) / (S,)
                
                # Add bias for this head: (S,) -> (B, D, S)
                x_h = x_h + self.bias[h, :S].unsqueeze(0).unsqueeze(0)
                
                # Put back
                x[:, h, :, :] = x_h
        
        # Reshape back to (B, E, S)
        x = x.reshape(B, E, S)
        
        # Transpose to (B, S, E) for output projection
        x = x.transpose(1, 2)
        
        # Apply output projection: (B, S, E) -> (B, S, E)
        if self.do_toep_proj:
            x = self.out_proj(x)
        
        return x
    
    @torch.inference_mode()
    def step(
        self,
        x_t: torch.Tensor,  # (B, E) - single timestep input
        hist: torch.Tensor | None,  # (B, H, D, Tprev) or None - history per head
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cached incremental forward pass for autoregressive generation.
        
        Args:
            x_t: Input features for current timestep, shape (B, E)
            hist: History of features per head, shape (B, H, D, Tprev) or None
            max_len: Maximum history length to maintain
            
        Returns:
            y_t: Output for current timestep, shape (B, E)
            new_hist: Updated history, shape (B, H, D, T)
        """
        B, E = x_t.shape
        H = self.n_heads
        D = self.hidden_dim
        
        # Apply input projection if needed
        if self.do_toep_proj:
            x_t = self.inp_proj(x_t)  # (B, E)
        
        # Reshape to (B, H, D)
        x_t = x_t.view(B, H, D)
        
        # Update history
        if hist is None:
            hist = x_t.unsqueeze(-1)  # (B, H, D, 1)
        else:
            hist = torch.cat([hist, x_t.unsqueeze(-1)], dim=-1)  # (B, H, D, T)
        
        # Trim history if needed
        if hist.size(-1) > max_len:
            hist = hist[:, :, :, -max_len:].contiguous()
        
        T = hist.size(-1)
        
        # Compute output for each head
        # For each head h, we want: y[h] = sum_{t'=0}^{T-1} w[h, T-1-t'] * hist[h, :, t']
        # This is equivalent to: y[h] = hist[h, :, :] @ flip(w[h, :T])
        
        w = self.weight[:, :T]  # (H, T)
        w_flipped = torch.flip(w, dims=[1])  # (H, T)
        
        # Compute: (B, H, D, T) * (H, T) -> (B, H, D)
        # Use einsum: bhdt,ht->bhd
        y_t = torch.einsum("bhdt,ht->bhd", hist, w_flipped)
        
        # Apply normalization if enabled
        if self.do_toep_mean:
            norm_factors = torch.cumsum(self.weight[:, :T], dim=-1)  # (H, T)
            norm_factor_current = norm_factors[:, -1]  # (H,) - use the last (current) position
            y_t = y_t / norm_factor_current.view(1, H, 1)  # (B, H, D) / (1, H, 1)
        
        # Add bias for current timestep
        bias_current = self.bias[:, T-1]  # (H,) - bias for position T-1
        y_t = y_t + bias_current.view(1, H, 1)  # (B, H, D) + (1, H, 1)
        
        # Reshape back to (B, E)
        y_t = y_t.reshape(B, E)
        
        # Apply output projection if needed
        if self.do_toep_proj:
            y_t = self.out_proj(y_t)  # (B, E)
        
        return y_t, hist


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

    def forward(
        self,
        x: torch.Tensor,
        parallel: bool
        ) -> torch.Tensor:

        x = x + self.mixer(x=self.mixer_norm(x), parallel=parallel)
        x = x + self.mlp(self.mlp_norm(x))

        return x

class MLPMixer(nn.Module):

    def __init__(self, config: Config):

        super().__init__()

        # Input Embedding
        self.inp_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)])
        self.out_norm = nn.RMSNorm(config.embed_dim)  # Normalize before output projection
        self.out_embed = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, MultiHeadMixer) or isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.2)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        parallel: bool = True,
        **kwargs,  # Ignore other HF-specific arguments like attention_mask, token_type_ids
    ) -> Dict[str, torch.Tensor | None]:
        
        x = self.inp_embed(input_ids)
        for block in self.blocks:
            x = block(x, parallel)
        x = self.out_norm(x)  # Normalize before output projection
        logits = self.out_embed(x)

        loss = None
        if labels is not None:
            # Shift for causal language modeling:
            # - logits[:, :-1] predicts labels[:, 1:]
            # - We're predicting the NEXT token at each position
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
        return {"loss": loss, "logits": logits}
    
    @torch.inference_mode()
    def generate_old(
        self, 
        x: torch.Tensor, 
        max_tokens: int,
        parallel: bool = True
    ) -> torch.Tensor:
        """
        Un-cached greedy autoregressive generation (slow but simple).
        
        Args:
            x: Input token IDs, shape (B, T) or (T,)
            max_tokens: Number of tokens to generate
            parallel: Whether to use parallel head processing
            
        Returns:
            Generated token sequence, shape (B, T+max_tokens)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        seq_len = self.blocks[0].mixer.seq_len
        
        for _ in range(max_tokens):
            # Crop to context window
            x_cond = x[:, -seq_len:]
            
            # Forward pass
            output = self.forward(x_cond, parallel=parallel)
            logits = output["logits"]
            
            # Greedy sampling
            next_id = logits[:, -1, :].argmax(dim=-1)
            
            # Append
            x = torch.cat([x, next_id.unsqueeze(1)], dim=1)
        
        return x
    
    @torch.inference_mode()
    def generate(
        self, 
        x: torch.Tensor, 
        max_tokens: int,
        parallel: bool = True
    ) -> torch.Tensor:
        """
        Cached greedy autoregressive generation (fast).
        
        Uses cached incremental inference - only processes new tokens
        rather than reprocessing entire sequence each step.
        
        Args:
            x: Input token IDs, shape (B, T) or (T,)
            max_tokens: Number of tokens to generate
            parallel: Whether to use parallel head processing (note: step() always efficient)
            
        Returns:
            Generated token sequence, shape (B, T+max_tokens)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        seq_len = self.blocks[0].mixer.seq_len
        
        # ---- PREFILL PHASE ----
        # Crop to context window
        x = x[:, -seq_len:]
        
        # Run one forward pass to build per-block caches
        caches = []
        h = self.inp_embed(x)  # (B, T, E)
        
        for block in self.blocks:
            # Normalize and save as history
            n1 = block.mixer_norm(h)  # (B, T, E)
            
            # Convert to multi-head format: (B, T, E) -> (B, H, D, T)
            B, T, E = n1.shape
            H = block.mixer.n_heads
            D = block.mixer.hidden_dim
            
            # Apply input projection if needed
            if block.mixer.do_toep_proj:
                n1 = block.mixer.inp_proj(n1)
            
            # Reshape: (B, T, E) -> (B, E, T) -> (B, H, D, T)
            hist = n1.transpose(1, 2).contiguous()  # (B, E, T)
            hist = hist.view(B, H, D, T)  # (B, H, D, T)
            caches.append(hist)
            
            # Continue forward pass
            mix_full = block.mixer(block.mixer_norm(h), parallel=parallel)  # (B, T, E)
            h = h + mix_full
            n2 = block.mlp_norm(h)
            h = h + block.mlp(n2)
        
        h = self.out_norm(h)  # Normalize before output projection
        h = self.out_embed(h)  # (B, T, vocab_size)
        logits = h
        
        # ---- DECODE PHASE ----
        for _ in range(max_tokens):
            # Sample next token
            next_ids = logits[:, -1, :].argmax(dim=-1)  # (B,)
            x = torch.cat([x, next_ids.unsqueeze(1)], dim=1)
            
            # Single-step through blocks using cached inference
            h_t = self.inp_embed(next_ids)  # (B, E)
            new_caches = []
            
            for block, cache in zip(self.blocks, caches):
                # Normalize
                n1_t = block.mixer_norm(h_t)  # (B, E)
                
                # Cached mixer step
                y_t, new_hist = block.mixer.step(
                    n1_t, cache, seq_len
                )  # (B, E), (B, H, D, T)
                
                h_t = h_t + y_t
                n2_t = block.mlp_norm(h_t)
                h_t = h_t + block.mlp(n2_t)
                
                new_caches.append(new_hist)
            
            caches = new_caches
            
            # Get logits for next token
            h_t_norm = self.out_norm(h_t)  # Normalize before output projection
            logits_t = self.out_embed(h_t_norm).unsqueeze(1)  # (B, 1, vocab_size)
            logits = torch.cat([logits, logits_t], dim=1)
        
        return x