"""
Train and benchmark 4 model variants of MultiHeadMixer.

Trains all combinations of:
- do_toep_mean: [True, False]
- do_toep_proj: [True, False]

Then benchmarks each model with both parallel=True and parallel=False for generation.
"""

import torch
import time
import json
import os
from pathlib import Path
from typing import Dict
from datasets import load_from_disk
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from mixer import MLPMixer, Config
import shutil
from datetime import datetime


def get_model_name(do_toep_mean: bool, do_toep_proj: bool) -> str:
    """Generate a descriptive name for the model variant."""
    mean_str = "mean" if do_toep_mean else "nomean"
    proj_str = "proj" if do_toep_proj else "noproj"
    return f"{mean_str}_{proj_str}"


def create_model(
    vocab_size: int,
    do_toep_mean: bool,
    do_toep_proj: bool,
    device: torch.device,
    seq_len: int = 512,
    dim: int = 512,
    layers: int = 10,
    num_heads: int = 4,
) -> MLPMixer:
    """Create a model with specified configuration."""
    config = Config(
        vocab_size=vocab_size,
        embed_dim=dim,
        seq_len=seq_len,
        num_blocks=layers,
        num_heads=num_heads,
        mlp_dim=dim,
        dropout=0.1,
        do_toep_mean=do_toep_mean,
        do_toep_proj=do_toep_proj,
    )
    model = MLPMixer(config).to(device)
    print(f"Model params: {model.count_params():,}")
    return model


def train_model(
    model: MLPMixer,
    train_dataset,
    eval_dataset,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    num_epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
) -> Dict:
    """Train a model and return metrics."""
    print(f"\nTraining model, saving to: {output_dir}")
    
    # Calculate optimal number of workers based on available CPUs
    num_workers = min(max(os.cpu_count() - 1, 1), 64)
    print(f"Using {num_workers} DataLoader workers (available CPUs: {os.cpu_count()})")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        eval_steps=2000,
        save_steps=4000,
        learning_rate=learning_rate,
        bf16=True,
        eval_strategy="steps",
        optim="adamw_torch",
        overwrite_output_dir=True,
        save_safetensors=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        logging_steps=500,
        include_inputs_for_metrics=False,
        skip_memory_metrics=True,
        torch_compile=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train!
    model.train()
    train_result = trainer.train()
    
    # Get final evaluation metrics
    eval_metrics = trainer.evaluate()
    
    return {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_metrics["eval_loss"],
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
    }


@torch.no_grad()
def benchmark_generation(
    model: MLPMixer,
    tokenizer: AutoTokenizer,
    device: torch.device,
    parallel: bool,
    use_cache: bool = True,
    num_tokens: int = 128,
    num_runs: int = 10,
    batch_size: int = 4,
) -> Dict:
    """
    Benchmark generation speed for a model.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        device: Device
        parallel: Whether to use parallel head processing
        use_cache: If True, use cached generation (fast), else uncached (slow)
        num_tokens: Number of tokens to generate
        num_runs: Number of benchmark runs
        batch_size: Batch size for generation
    """
    model.eval()
    
    # Create prompt (just a simple start token repeated)
    prompt = torch.full((batch_size, 1), tokenizer.bos_token_id or 0, device=device, dtype=torch.long)
    
    times = []
    
    # Choose generation method
    gen_fn = model.generate if use_cache else model.generate_old
    
    # Warmup runs
    for _ in range(2):
        _ = gen_fn(prompt, num_tokens, parallel=parallel)
    
    # Benchmark runs
    for run in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        output = gen_fn(prompt, num_tokens, parallel=parallel)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    tokens_per_second = (num_tokens * batch_size) / avg_time
    
    return {
        "parallel": parallel,
        "use_cache": use_cache,
        "avg_time_seconds": avg_time,
        "tokens_per_second": tokens_per_second,
        "num_tokens": num_tokens,
        "batch_size": batch_size,
        "num_runs": num_runs,
    }


def main():
    """Main training and benchmarking loop."""
    
    # Setup
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    
    # Load datasets
    print("\nLoading datasets...")
    data_root = Path(__file__).parent.parent.parent / "data" / "SimpleStories" / "data"
    train_dataset = load_from_disk(str(data_root / "train"))
    eval_dataset = load_from_disk(str(data_root / "test"))
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Eval examples: {len(eval_dataset):,}")
    
    # Setup output directory
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(__file__).parent / "results" / date_str
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the training script for reproducibility
    code_path = Path(__file__).resolve()
    shutil.copy(code_path, results_dir / "train_script.py")
    shutil.copy(code_path.parent / "mixer.py", results_dir / "model_snapshot.py")
    
    # Define model variants to train
    variants = [
        (True, True),    # mean + proj
        (True, False),   # mean, no proj
        (False, True),   # no mean + proj
        (False, False),  # no mean, no proj
    ]
    
    all_results = []
    
    # Train and benchmark each variant
    for do_toep_mean, do_toep_proj in variants:
        model_name = get_model_name(do_toep_mean, do_toep_proj)
        print("\n" + "="*80)
        print(f"TRAINING VARIANT: {model_name}")
        print(f"  do_toep_mean: {do_toep_mean}")
        print(f"  do_toep_proj: {do_toep_proj}")
        print("="*80)
        
        # Create model
        model = create_model(
            vocab_size=vocab_size,
            do_toep_mean=do_toep_mean,
            do_toep_proj=do_toep_proj,
            device=device,
        )
        
        # Train
        output_dir = results_dir / model_name
        training_metrics = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )
        
        # Benchmark generation: cached/uncached × parallel/sequential = 4 combinations
        print(f"\nBenchmarking generation for {model_name}...")
        
        # Cached + Parallel
        print("  Testing cached + parallel...")
        cached_parallel = benchmark_generation(
            model=model, tokenizer=tokenizer, device=device,
            parallel=True, use_cache=True,
        )
        
        # Cached + Sequential  
        print("  Testing cached + sequential...")
        cached_sequential = benchmark_generation(
            model=model, tokenizer=tokenizer, device=device,
            parallel=False, use_cache=True,
        )
        
        # Uncached + Parallel
        print("  Testing uncached + parallel...")
        uncached_parallel = benchmark_generation(
            model=model, tokenizer=tokenizer, device=device,
            parallel=True, use_cache=False,
        )
        
        # Uncached + Sequential
        print("  Testing uncached + sequential...")
        uncached_sequential = benchmark_generation(
            model=model, tokenizer=tokenizer, device=device,
            parallel=False, use_cache=False,
        )
        
        # Collect results
        result = {
            "model_name": model_name,
            "config": {
                "do_toep_mean": do_toep_mean,
                "do_toep_proj": do_toep_proj,
            },
            "training": training_metrics,
            "generation": {
                "cached_parallel": cached_parallel,
                "cached_sequential": cached_sequential,
                "uncached_parallel": uncached_parallel,
                "uncached_sequential": uncached_sequential,
            },
            "speedups": {
                "cache_speedup_parallel": cached_parallel["tokens_per_second"] / uncached_parallel["tokens_per_second"],
                "cache_speedup_sequential": cached_sequential["tokens_per_second"] / uncached_sequential["tokens_per_second"],
                "parallel_speedup_cached": cached_parallel["tokens_per_second"] / cached_sequential["tokens_per_second"],
                "parallel_speedup_uncached": uncached_parallel["tokens_per_second"] / uncached_sequential["tokens_per_second"],
            }
        }
        all_results.append(result)
        
        # Save individual result
        result_path = output_dir / "metrics.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved metrics to: {result_path}")
        
        # Print summary
        print(f"\n{model_name} Summary:")
        print(f"  Eval Loss: {training_metrics['eval_loss']:.4f}")
        print(f"  Generation Speed:")
        print(f"    Cached + Parallel:     {cached_parallel['tokens_per_second']:>8.1f} tokens/s")
        print(f"    Cached + Sequential:   {cached_sequential['tokens_per_second']:>8.1f} tokens/s")
        print(f"    Uncached + Parallel:   {uncached_parallel['tokens_per_second']:>8.1f} tokens/s")
        print(f"    Uncached + Sequential: {uncached_sequential['tokens_per_second']:>8.1f} tokens/s")
        print(f"  Speedups:")
        print(f"    Cache (parallel):   {result['speedups']['cache_speedup_parallel']:>6.2f}x")
        print(f"    Cache (sequential): {result['speedups']['cache_speedup_sequential']:>6.2f}x")
        print(f"    Parallel (cached):  {result['speedups']['parallel_speedup_cached']:>6.2f}x")
        print(f"    Parallel (uncached):{result['speedups']['parallel_speedup_uncached']:>6.2f}x")
        
        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save combined results
    combined_path = results_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print final comparison table
    print("\n" + "="*120)
    print("FINAL RESULTS SUMMARY")
    print("="*120)
    print(f"{'Model':<20} {'Eval Loss':<12} {'Cached+Par':<14} {'Cached+Seq':<14} {'Uncached+Par':<14} {'Cache Speedup':<15}")
    print("-"*120)
    
    for result in all_results:
        gen = result['generation']
        speedup = result['speedups']['cache_speedup_parallel']
        print(
            f"{result['model_name']:<20} "
            f"{result['training']['eval_loss']:<12.4f} "
            f"{gen['cached_parallel']['tokens_per_second']:<14.1f} "
            f"{gen['cached_sequential']['tokens_per_second']:<14.1f} "
            f"{gen['uncached_parallel']['tokens_per_second']:<14.1f} "
            f"{speedup:<15.2f}x"
        )
    
    print("\n" + "="*120)
    print("SPEEDUP ANALYSIS")
    print("="*120)
    print(f"{'Model':<20} {'Cache (par)':<15} {'Cache (seq)':<15} {'Parallel (cached)':<20} {'Parallel (uncached)':<20}")
    print("-"*120)
    
    for result in all_results:
        sp = result['speedups']
        print(
            f"{result['model_name']:<20} "
            f"{sp['cache_speedup_parallel']:<15.2f}x "
            f"{sp['cache_speedup_sequential']:<15.2f}x "
            f"{sp['parallel_speedup_cached']:<20.2f}x "
            f"{sp['parallel_speedup_uncached']:<20.2f}x"
        )
    
    print("\n" + "="*120)
    print(f"All results saved to: {combined_path}")
    print("="*120)


if __name__ == "__main__":
    main()
