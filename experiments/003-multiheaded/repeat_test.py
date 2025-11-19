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
# define a MLP Mixer based causal-language-model using weight masking

class RepeatCausalLinear(nn.Module):

    def __init__(self, dim: int):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        [ a  a  a  a ]
        [ 0  b  b  b ]
        [ 0  0  c  c ]
        [ 0  0  0  d ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        M = torch.where(
            j >= i, v[i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
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


class RepeatHeads(nn.Module):

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
                [ToeplitzCausalLinear(seq_len) for i in range(n_heads)]
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


class MixerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        expansion_factor: int = 4,
        dropout: float = 0.,
        heads=None,
        expanded_convs=False,
    ):

        super(MixerBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor

        # channel-norm
        self.channel_norm = nn.LayerNorm(hidden_dim)

        # channel-mixing layer
        self.channel_mixing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

        # token-norm
        self.token_norm = nn.LayerNorm(hidden_dim)
        if heads and heads > 0:
            self.token_mixing_layer = ToeplitzHeads(
                hidden_dim,
                seq_len,
                hidden_dim // heads,
                heads,
                expanded_convs=expanded_convs,
            )  # type: ignore[assignment]

        else:

            if expanded_convs:
                # token-mixing layer
                self.token_mixing_layer = nn.Sequential(
                    ToeplitzCausalLinear(seq_len),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    ToeplitzCausalLinear(seq_len),
                )  # type: ignore[assignment]

            else:
                # flat mixer layer
                self.token_mixing_layer = RepeatCausalLinear(seq_len)  # type: ignore[assignment]

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
        expanded_convs=False,
        copy=False,
        tie_io=False,
    ):

        super(MLPMixer, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks

        # Input Embedding
        self.input_layer = nn.Embedding(vocab_size, hidden_dim)

        # Mixer Blocks
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(
                    hidden_dim, seq_len, heads=heads, expanded_convs=expanded_convs
                )
                for _ in range(num_blocks)
            ]
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
        self.copy = copy

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, RepeatCausalLinear):
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

    tokenized_length = 1024
    dim = 128
    layers = 16
    n_heads = None

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False, copy=True
    ).float()
    n_gpus = torch.cuda.device_count()
    total_batch_size = 128
    batch_size = total_batch_size // n_gpus
    train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"
    output_dir = f"{checkpoint_root}/fineweb_h{n_heads}_repeat_{dim}_n{layers}_c512_b{batch_size}x{n_gpus}"

    
    datasets.config.IN_MEMORY_MAX_SIZE = 50e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None).filter(lambda x: x['input_ids'][-1] != 1).take(5000)
    print(len(train_dataset), len(test_dataset))
    mlflow.end_run()
    print("training begun")
    print(model)
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        eval_steps=100,
        save_steps=10000,
        learning_rate=5e-4,
        bf16=True,
        eval_strategy="steps",
        output_dir=output_dir,
        optim="adamw_torch",
        overwrite_output_dir=True,
        save_safetensors=True,
        max_steps=10000,
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
