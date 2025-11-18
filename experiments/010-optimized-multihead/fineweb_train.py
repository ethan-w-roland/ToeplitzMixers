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
from mixer import MLPMixer, Config
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    vocab_size = n_vocab
    tokenized_length = 512
    dim = 512
    layers = 16
    n_heads = 4

    config = Config(
            vocab_size=vocab_size,
            embed_dim=dim,
            seq_len=tokenized_length,
            num_heads=n_heads,
            mlp_dim=dim*4,
            dropout=0.,
            do_toep_mean=False,
            do_toep_proj=True,
            parallel_mixer=True,
            num_blocks=layers
        )

    model = MLPMixer(config).float()

    train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

    total_batch_size = 128
    n_gpus = torch.cuda.device_count()
    batch_size = total_batch_size // n_gpus

    output_dir = f"{checkpoint_root}/fineweb_toep_parallel_h{n_heads}_{dim}_n{layers}_b{batch_size}x{n_gpus}"

    datasets.config.IN_MEMORY_MAX_SIZE = 50e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None)
    print(len(train_dataset), len(test_dataset))
    mlflow.end_run()
    print("training begun")
    print(model)
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=8000,
        learning_rate=5e-4,
        bf16=True,
        eval_strategy="steps",
        output_dir=output_dir,
        optim="adamw_torch",
        overwrite_output_dir=True,
        save_safetensors=True,
        max_steps=200000,
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
