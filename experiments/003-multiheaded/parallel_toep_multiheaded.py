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

class ToeplitzCausalLinear(nn.Module):
    """
    A linear layer with a triangular (causal) mask applied to the weight matrix.
    This ensures each position i cannot use info from positions > i.
    """

    def __init__(self, dim: int):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

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

class HeadedToeplitzCausalLinear(nn.Module):
    """
    Multi-headed parallelized implementation
    """

    def __init__(self, dim: int, heads: int):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
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
        x = x.to(device)
        W = self.vector_to_matrix(self.weight).repeat(x.shape[0]//self.heads, 1, 1) 
        output = torch.bmm(x, W)
        repeated_bias = self.bias.repeat(x.shape[0]//self.heads, 1)
        repeated_bias = repeated_bias.unsqueeze(1).repeat(1, x.shape[1], 1)
        output += repeated_bias
        return output

class ToeplitzHeads(nn.Module):

    def __init__(
        self,
        dim: int,
        seq_len: int,
        hidden_dim: int,
        n_heads: int,
        expanded_convs: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.proj_head = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for i in range(n_heads)]
        ).to(device)

        self.out_proj = nn.Linear(dim, dim)

        if expanded_convs:
            self.mixer_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        ToeplitzCausalLinear(seq_len),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        ToeplitzCausalLinear(seq_len),
                    )
                    for i in range(n_heads)
                ]
            )
        else:
            self.mixer_heads = nn.ModuleList(
                [RepeatCausalLinear(seq_len) for i in range(n_heads)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        x = rearrange(x, "b e t -> b t e")
        # pre-concatenated out projection
        for head in range(self.n_heads):
            projection = self.proj_head[head](x)
            projection = rearrange(projection, "b t e -> b e t")
            conv_projection = self.mixer_heads[head](projection)
            rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
            activations.append(rearranged_conv)

        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=2)
        hidden_layer = self.out_proj(hidden_layer)
        hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
        return hidden_layer
    
class ParallelToeplitzHeads(nn.Module):

    def __init__(
        self,
        dim: int,
        seq_len: int,
        n_heads: int,
    ):
        # note that the hidden dim is by definition dim // n_heads
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.mixer_heads = HeadedToeplitzCausalLinear(seq_len, n_heads)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b e t -> b t e")
        headed_projection = self.in_proj(x) 
        projections = rearrange(headed_projection, "b t (h e) -> (b h) e t", h=self.n_heads)
        conv_projection = self.mixer_heads(projections)
        rearranged_conv = rearrange(conv_projection, "(b h) e t -> b t (h e)", h=self.n_heads)
        output = self.out_proj(rearranged_conv)
        output = rearrange(output, "b t e -> b e t")
        return output

class MixerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        expansion_factor: int = 4,
        dropout: float = 0.,
        heads=None
    ):

        super(MixerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor

        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mixing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

        self.token_norm = nn.LayerNorm(hidden_dim)
        if heads and heads > 0:
            self.token_mixing_layer = ParallelToeplitzHeads(
                hidden_dim,
                seq_len,
                heads
            ) 

        else:
            self.token_mixing_layer = ToeplitzCausalLinear(seq_len)  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class MLPMixer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        copy=False
    ):
        super(MLPMixer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks

        self.input_layer = nn.Embedding(vocab_size, hidden_dim)
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(
                    hidden_dim, seq_len, heads=heads
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.copy = copy

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, ToeplitzCausalLinear) or isinstance(m, HeadedToeplitzCausalLinear):
                # Kaiming He initialization for Swish activation
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None, **kwargs):
        if self.copy:
            input_ids = copy_dataset(input_ids)
            if labels is not None:
                labels = copy_dataset(labels)
        labels = labels[:, 1:].contiguous()
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x)
        logits = self.output_layer(x)
        logits = logits[:, :-1].contiguous()

        if labels is not None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits

def copy_dataset(input_ids):
    n_ctx = len(input_ids[0])
    for i, input in enumerate(input_ids):
        first_half = input[:n_ctx//2]
        copied_halves = torch.cat((first_half, first_half))
        input_ids[i] = copied_halves
    return input_ids

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    tokenized_length = 512
    dim = 256
    layers = 16
    n_heads = 4

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, copy=False
    ).float()

    train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"
    output_dir = f"{checkpoint_root}/fineweb_parallel_h4_toep_256_n16_c512_b64x2"
    
    datasets.config.IN_MEMORY_MAX_SIZE = 5e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None)
    print(len(train_dataset), len(test_dataset))
    mlflow.end_run()
    print("training begun")
    print(model)
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=8000,
        learning_rate=5e-4,
        fp16=True,
        eval_strategy="steps",
        output_dir=output_dir,
        optim="adamw_torch",
        overwrite_output_dir=True,
        save_safetensors=True,
        max_steps=200000,
        torch_compile=True
    )

    trainer = transformers.Trainer(
        model=model.to("cuda"),  # pre-assignment for FSDP initialization
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
