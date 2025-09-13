import torch

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
    k = 4
    p = k-1 # pad value
    B, E, S = x.shape
    W = vector_to_matrix(weight, k)
    print (W)
    # apply pad for k>1 convolution
    padded_x = torch.nn.functional.pad(input=x, pad=(0, 0, p, p), mode='constant', value=0)
    padded_e = padded_x.shape[1]
    accumulated_output = torch.zeros(x.shape)
    
    for i in range(k):
        print ('k=', i)
        end_index = E + i + p
        offset = p - i # indices to ignore due to padding
        print (padded_x.shape, W[i].shape)
        print (padded_x[:, i:end_index], padded_x[:, i:end_index].shape)
        out = padded_x[:, i:end_index] @ W[i]  # [B, padded_e, S] @ [S, S] -> [B, padded_e, S]
        print (out.shape)
        print (out[:, offset:E + offset])
        accumulated_output += out[:, offset:E + offset]

    accumulated_output += bias
    return accumulated_output

weight = torch.tensor([[1,2,3,4], [2,4,6,8]]).to(torch.float)
weight = torch.tensor([[1,2,3,4], [2,4,6,8], [3,4,5,6],[4,3,2,1]]).to(torch.float)
bias = torch.zeros(weight[1].shape)
x = torch.ones((1, 5, 4)).to(torch.float) # [b e t]
forward(x, weight, bias)