import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel
import datasets
from datasets import load_from_disk
import mlflow
import os
from dotenv import load_dotenv
import shutil

all_hammings = []
hamming_log =[]

class LlamaCLM(nn.Module):
    def __init__(self, model, dim, vocab_size, copy=True):
        super().__init__()
        self.model = model
        self.input_layer = nn.Embedding(vocab_size, dim)
        self.output_layer = nn.Linear(dim, vocab_size)
        self.copy = copy
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        if self.copy:
            input_ids = copy_dataset(input_ids)
            attention_mask = copy_dataset(attention_mask)
            if labels is not None:
                labels = copy_labels(labels) # masks first half
        labels = labels[:, 1:].contiguous()
        x = self.model(input_ids, attention_mask=attention_mask)
        logits = self.output_layer(x.last_hidden_state)
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

def init_llama():
    decoder_dim = 256
    context_length = 1024
    n_layers = 16
    n_heads = 4

    vocab_size = 8000
    llama_config_kwargs = {
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': n_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size
    }

    # Initializing a LLaMA model
    configuration = LlamaConfig(**llama_config_kwargs)

    # Initializing a model from the llama-7b style configuration
    model = LlamaModel(configuration).float()
    return (model)

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
    n_heads = 4

    model = init_llama()
    model = LlamaCLM(model, dim, n_vocab, copy=True)

    train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"

    output_dir = f"{checkpoint_root}/fineweb_copy_llama_512_n16_c1024_b16x4"

    
    datasets.config.IN_MEMORY_MAX_SIZE = 50e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None)
    print(len(train_dataset), len(test_dataset))
    mlflow.end_run()
    print("training begun")
    print(model)
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        eval_steps=100,
        save_steps=10000,
        learning_rate=2e-4,
        fp16=True,
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
