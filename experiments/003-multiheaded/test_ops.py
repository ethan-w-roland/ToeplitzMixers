import torch
from einops import rearrange
import torch.nn as nn

"""
Tests equivalence of parallelized multi-headed mixer operation (via fully materialized weight matmult and bias add) with sequential mat mult, ignoring
input and output projections
"""

class HeadedToeplitzCausalLinear(nn.Module):
    """
    Multi-headed parallelized implementation
    """

    def __init__(self, dim: int, heads: int):

        super().__init__()

        # Standard weight + bias
        self.weight = WEIGHT
        self.bias = BIAS
        self.heads = heads

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        Given a matrix v of shape (k, m) and head number h >= 0, 
        returns an (k x m x m) matrix M where M[i, j] = v[j - i] if 
        j >= i, and 0 otherwise.

        For example, if v = [[a, b, c, d], [e, f, g, h]], k=2 then M will be:

        [[
            [ a  b  c  d ]
            [ 0  a  b  c ]
            [ 0  0  a  b ]
            [ 0  0  0  a ]
        ],
        [
            [ e  f  g  h ]
            [ 0  e  f  g ]
            [ 0  0  e  f ]
            [ 0  0  0  e ]
        ]]

        """
        # Expects v is a preformed tensor with shape [k, D]
        m = v.shape[-1] # vector shape

        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        # j - i gives the offset into v. When j < i, we want a 0.
        M = torch.where(
            j >= i, v[..., j - i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
        )
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.vector_to_matrix(self.weight)
        W = W.repeat(x.shape[0]//self.heads, 1, 1) 
        output = torch.bmm(x, W)
        repeated_bias = self.bias.repeat(x.shape[0]//self.heads, 1)
        repeated_bias = repeated_bias.unsqueeze(1).repeat(1, x.shape[1], 1)
        output += repeated_bias
        return output

    def _interleave_forward(self, x):
        # note that this scrambles the batch/head order for the weight matrix matmult 
        # w/ rearrange (b h) e t -> b t (h e)

        W = self.vector_to_matrix(self.weight)
        W = W.repeat_interleave(x.shape[0]//self.heads, dim=0)
        output = torch.bmm(x, W)
        repeated_bias = self.bias.repeat_interleave(x.shape[0]//self.heads, dim=0)
        repeated_bias = repeated_bias.unsqueeze(1).repeat(1, x.shape[1], 1)
        output += repeated_bias
        return output

class ToeplitzCausalLinear(nn.Module):
    """
    A linear layer with a triangular (causal) mask applied to the weight matrix.
    This ensures each position i cannot use info from positions > i.
    """

    def __init__(self, dim: int, weight, bias):

        super().__init__()

        # Standard weight + bias
        self.weight = weight
        self.bias = bias

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
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
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        # j - i gives the offset into v. When j < i, we want a 0.
        M = torch.where(
            j >= i, v[j - i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
        )
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, embed_dim, seq_len)
        """
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out


def forward(x: torch.Tensor,) -> torch.Tensor:
    mixer_heads = nn.ModuleList(
                [ToeplitzCausalLinear(6, WEIGHT[i], BIAS[i]) for i in range(HEADS)]
            )

    activations = []
    x = rearrange(x, "b e t -> b t e")
    size = 3
    # pre-concatenated out projection
    for head in range(HEADS): #three heads
        start, end = head*size, head*size + size
        projection = x[:, :, start:end] # identity projection
        projection = rearrange(projection, "b t e -> b e t")
        conv_projection = mixer_heads[head](projection)
        rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
        activations.append(rearranged_conv)

    # concatenate and project multi-headed output
    hidden_layer = torch.cat(activations, dim=2)
    hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
    return hidden_layer


def parallel_forward(x) -> torch.Tensor:
    mixer_heads = HeadedToeplitzCausalLinear(6, 3)
    x = rearrange(x, "b e t -> b t e")
    projections = rearrange(x, "b t (h e) -> (b h) e t", h=3)
    conv_projection = mixer_heads(projections)
    rearranged_conv = rearrange(conv_projection, "(b h) e t -> b t (h e)", h=HEADS)
    output = rearrange(rearranged_conv, "b t e -> b e t")
    return output

HEADS = 3
WEIGHT = nn.Parameter(torch.randn(HEADS, 6))
BIAS = nn.Parameter(torch.zeros(HEADS, 6))
x = torch.randn((32, 9, 6)).to(torch.float) # [b e t] shape
pout = parallel_forward(x)
out = forward(x)

# Test that parallelized head op is identical to sequential head op
assert torch.allclose(pout, out, rtol=1e-3)