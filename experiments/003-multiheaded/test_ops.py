import torch
from einops import rearrange

class ToeplitzHeads(nn.Module):

    def __init__(
        self,
        dim: int,
        seq_len: int,
        hidden_dim: int,
        n_heads: int,
        expanded_convs: bool = False,
        dropout: float = 0.,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.proj_head = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for i in range(n_heads)]
        ).to(device)

        self.out_proj = nn.Linear(dim, dim)

        self.mixer_heads = MultiToeplitzCausalLinear(seq_len, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        x = rearrange(x, "b e t -> b t e")

        headed_projection = self.head_projections(x)
        projections = rearrange(projection, "b t e -> b h e t", h=self.n_heads)
        conv_projections = self.mixer_heads(projections)
        rearranged_conv = rearrange(conv_projection, "b h e t -> b e t", h=self.n_heads)


        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=2)
        hidden_layer = self.out_proj(hidden_layer)
        hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
        return hidden_layer
        
def mvector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
    """
    Given a matrix v of shape (k, m) and convolutional kernel k >= 0, 
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
    x = x.to(device)
    B, E, S = x.shape
    W = self.vector_to_matrix(self.weight)
    out = processed_x @ W + self.bias
    return out

def vector_to_matrix(v: torch.Tensor, k: int) -> torch.Tensor:
    """
    Given a vector v of shape (m,), returns an (m x m) matrix M
    where M[i, j] = v[j - i] if j >= i, and 0 otherwise.

    For example, if v = [a, b, c, d] then M will be:

    [ a  b  c  d ]
    [ 0  a  b  c ]
    [ 0  0  a  b ]
    [ 0  0  0  a ]
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


def forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x shape: (batch, embed_dim, seq_len)
    """
    ctivations = []
    x = rearrange(x, "b e t -> b t e")
    size = 2
    # pre-concatenated out projection
    for head in range(3): #three heads
        start, end = head*size, head*size + size
        projection = x[:, :, start:end] # mock proj
        projection = rearrange(projection, "b t e -> b e t")
        conv_projection = mixer_heads[head](projection)
        rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
        activations.append(rearranged_conv)

    # concatenate and project multi-headed output
    hidden_layer = torch.cat(activations, dim=2)
    hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
    return hidden_layer


def parallel_forward(x, weight, bias):
    activations = []
    x = rearrange(x, "b e t -> b t e")
   
    headed_projection = x # mock in projection
    projections = rearrange(projection, "b t e -> b h e t", h=self.n_heads)
    conv_projections = self.mixer_heads(projections)
    rearranged_conv = rearrange(conv_projection, "b h e t -> b e t", h=self.n_heads)
    hidden_layer =  hidden_layer 
    hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
    return hidden_layer


# weight = torch.tensor([[1,2,3,4], [2,4,6,8]]).to(torch.float)
weight = torch.tensor([[1,2,3,4], [2,4,6,8], [3,4,5,6],[4,3,2,1]]).to(torch.float)
bias = torch.zeros(weight[1].shape)
x = torch.randn((1, 6, 4)).to(torch.float) # [b e t]
pout = parallel_forward(x, weight, bias)
out = forward(x, weight, bias)
assert torch.allclose(pout, out)