import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
import datasets
from datasets import load_from_disk
import mlflow
import os
from dotenv import load_dotenv
import shutil
from hyena_trainer_fineweb import HyenaModule

all_hammings = []
hamming_log =[]


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
            [HyenaModule(
                hidden_dim, 
                seq_len, 
                )
            for i in range(num_blocks)]
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
            if isinstance(m, nn.Linear):
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
                labels = copy_labels(labels) # masks first half
        
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x)

        logits = self.output_layer(x)
        labels = labels[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()

        global all_hammings
        if not self.training:
            all_hammings.append(hamming(logits, labels))
        if self.training and all_hammings: 
            print (f'Accuracy: {sum(all_hammings)/ len(all_hammings)}')
            global hamming_log; hamming_log.append(sum(all_hammings)/ len(all_hammings))
            all_hammings = []

        if labels is not None:
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits

@torch.no_grad()
def hamming(model_output, labels):
    total_metric = 0
    #ignore_list = [tokenizer.pad_token, tokenizer.encode(tokenizer.eos_token)[-1]]
    input_tokens = labels
    generated_tokens = torch.argmax(model_output, dim=-1)
    nonpad_tokens = torch.where(labels != -100, 1, 0)
    equal_tokens = torch.where(generated_tokens == labels, 1, 0) & nonpad_tokens
    average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
    return average_metric

@torch.no_grad()
def hamming_eval(model_output, labels):
    total_metric = 0 
    input_tokens = labels
    generated_tokens = torch.argmax(model_output, dim=-1)
    for i in range(len(generated_tokens)): # len(generated_tokens)
        # expects tokens to be pre-flattened
        assert len(input_tokens[i]) == len(generated_tokens[i])
        count, card = 0, 0
        pad_token = tokenizer.encode(tokenizer.pad_token)[-1] # will be [2]
        for j in range(len(input_tokens[i])//2, len(input_tokens[i])): # starts at the half way point  
            if input_tokens[i][j] == pad_token or input_tokens[i][j] == tokenizer.encode(tokenizer.eos_token)[-1]:
                continue
            else:
                card += 1
                if input_tokens[i][j] in generated_tokens[i][j]:
                    count += 1
        total_metric += (card - count) / card
    average_metric = torch.tensor([total_metric / len(generated_tokens)])
    return average_metric

def copy_dataset(input_ids):
    n_ctx = len(input_ids[0])
    for i, input in enumerate(input_ids):
        first_half = input[:n_ctx//2]
        copied_halves = torch.cat((first_half, first_half))
        input_ids[i] = copied_halves
    return input_ids

def copy_labels(labels):
    n_ctx = len(labels[0])
    for i, input in enumerate(labels):
        first_half = input[:n_ctx//2]
        pad_half = torch.ones(first_half.shape).to(device) * -100
        halves = torch.cat((pad_half, first_half))
        labels[i] = halves
    return labels
    
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
    dim = 256
    layers = 16
    n_heads = None

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False, copy=True
    ).float()

    #model = FrozenMixer(
    #   n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False, copy=True
    #).float()

    total_batch_size = 64
    n_gpus = torch.cuda.device_count()
    batch_size = total_batch_size // n_gpus

    train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"
    
    output_dir = f"{checkpoint_root}/fineweb_copy_hyena_{dim}_n{layers}_b{batch_size}x{n_gpus}"
    datasets.config.IN_MEMORY_MAX_SIZE = 50e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None).filter(lambda x: x['input_ids'][-1] != 1).take(5000)
    print(len(train_dataset), len(test_dataset))
    # print (test_dataset[0])
    mlflow.end_run()
    print("training begun")
    print(model)
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #gradient_accumulation_steps=2,
        warmup_steps=50,
        eval_steps=100,
        save_steps=10000,
        learning_rate=5e-4,
        fp16=True,
        eval_strategy="steps",
        output_dir=output_dir,
        optim="adamw_torch",
        overwrite_output_dir=True,
        save_safetensors=False,
        max_steps=10000,
    )

    trainer = transformers.Trainer(
        model=model.to("cuda"),  # pre-assignment for FSDP initialization
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print (f'Output path: {output_dir}')
    # save driver code snapshot in checkpoint dir 
    code_path = os.path.abspath(__file__) 
    if not os.path.isdir(output_dir): 
        os.mkdir(output_dir) 
    shutil.copy(code_path, output_dir) 

    model.train()
    trainer.train()
    print (hamming_log)
