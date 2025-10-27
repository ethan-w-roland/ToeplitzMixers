# Experiment 010: Optimized Multi-Head Toeplitz Mixer

## Overview

This experiment trains and benchmarks **4 variants** of the MultiHeadMixer model to understand the impact of two architectural choices:

1. **`do_toep_mean`**: Whether to normalize by cumulative sum of weights (frames mixing as weighted average)
2. **`do_toep_proj`**: Whether to use input/output projections in the mixing layer

Each model is then benchmarked with both **parallel** and **sequential** head processing to measure generation speed differences.

## Model Variants

| Model Name      | do_toep_mean | do_toep_proj | Description |
|-----------------|--------------|--------------|-------------|
| `mean_proj`     | ✅ True      | ✅ True      | Full model with normalization and projections |
| `mean_noproj`   | ✅ True      | ❌ False     | Normalization but no projections |
| `nomean_proj`   | ❌ False     | ✅ True      | Projections but no normalization |
| `nomean_noproj` | ❌ False     | ❌ False     | Minimal model - no normalization or projections |

## Metrics Collected

For each variant, we collect:

### Training Metrics
- **Train Loss**: Final training loss
- **Eval Loss**: Validation loss (best checkpoint)
- **Training Runtime**: Total training time
- **Samples per Second**: Training throughput

### Generation Benchmarks
- **Parallel Mode**: All heads processed simultaneously
  - Tokens per second
  - Average generation time
- **Sequential Mode**: Heads processed one at a time
  - Tokens per second
  - Average generation time
- **Speedup**: Ratio of parallel vs sequential speed

## Usage

### 1. Prepare Data

First, prepare the SimpleStories dataset:

```bash
cd ../../data/SimpleStories
python dataprep.py --max_length 512
```

This creates tokenized datasets in `data/train/` and `data/test/`.

### 2. Train All Variants

Run the main training script:

```bash
python train.py
```

This will:
- Train all 4 model variants sequentially
- Save checkpoints to `results/{model_name}/`
- Benchmark generation speed for each
- Save metrics to `results/all_results.json`

**Expected Runtime**: ~2-4 hours per model (depending on GPU)

### 3. Analyze Results

After training completes, analyze the results:

```bash
python analyze_results.py
```

This generates:
- Detailed summary of all metrics
- Comparison plots (`results/comparison_plots.png`)
- Identification of best models by different criteria

## Results Structure

```
results/
├── mean_proj/
│   ├── checkpoint-5000/
│   ├── checkpoint-10000/
│   ├── metrics.json
│   ├── train_script.py       # Snapshot of training code
│   └── model_snapshot.py     # Snapshot of model code
├── mean_noproj/
│   └── ...
├── nomean_proj/
│   └── ...
├── nomean_noproj/
│   └── ...
├── all_results.json          # Combined results
└── comparison_plots.png      # Visualization
```

## Key Findings to Investigate

After running the experiment, analyze:

1. **Quality Trade-offs**
   - Does normalization (`do_toep_mean`) improve eval loss?
   - Do projections (`do_toep_proj`) help model quality?

2. **Speed Trade-offs**
   - Which variant is fastest for generation?
   - How much overhead do projections add?
   - Is the parallel speedup consistent across variants?

3. **Optimal Configuration**
   - Which variant offers the best quality-speed balance?
   - Are there diminishing returns from added complexity?

## Model Architecture

### Base Configuration
```python
- Vocabulary Size: ~50k (SimpleStories tokenizer)
- Embedding Dimension: 512
- Sequence Length: 512
- Number of Blocks: 10
- Number of Heads: 4
- MLP Dimension: 512
- Dropout: 0.1
```

### MultiHeadMixer Variants

The `do_toep_mean` and `do_toep_proj` flags control:

**`do_toep_mean=True`:**
```python
# Normalize by cumulative sum of weights
norm_factors = torch.cumsum(self.weight, dim=-1)
x = x / norm_factors
```

**`do_toep_proj=True`:**
```python
# Add learnable projections
self.inp_proj = nn.Linear(embed_dim, embed_dim)
self.out_proj = nn.Linear(embed_dim, embed_dim)
```

## Training Details

### Lessons from Ben's Approach

This training script incorporates several best practices:

1. **HuggingFace Trainer API**
   - Automatic checkpointing and evaluation
   - Built-in FP16 training
   - Learning rate scheduling with warmup
   - Best model selection based on eval loss

2. **Code Snapshotting**
   - Saves training script and model code with checkpoints
   - Ensures reproducibility

3. **Comprehensive Metrics**
   - Training and validation loss
   - Generation speed benchmarks (cached vs uncached)
   - Throughput measurements

4. **Resource Management**
   - `save_total_limit=2` to avoid filling disk
   - Memory cleanup between model variants
   - `torch.cuda.empty_cache()` between runs

### Training Arguments

```python
- Learning Rate: 5e-4
- Batch Size: 16 per device
- Warmup Steps: 500
- Max Steps: 20,000
- FP16: Enabled
- Optimizer: AdamW (torch)
- Eval Strategy: Every 2,000 steps
- Save Strategy: Every 5,000 steps
```

## Customization

### Adjust Training

To modify training settings, edit `train.py`:

```python
training_metrics = train_model(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    output_dir=output_dir,
    num_epochs=2,           # ← Change here
    batch_size=16,          # ← Change here
    learning_rate=5e-4,     # ← Change here
    max_steps=20000,        # ← Change here
)
```

### Adjust Model Architecture

To modify model architecture:

```python
model = create_model(
    vocab_size=vocab_size,
    do_toep_mean=do_toep_mean,
    do_toep_proj=do_toep_proj,
    device=device,
    seq_len=512,      # ← Change here
    dim=512,          # ← Change here
    layers=10,        # ← Change here
    num_heads=4,      # ← Change here
)
```

### Adjust Benchmarking

To modify generation benchmarks:

```python
benchmark_metrics = benchmark_generation(
    model=model,
    tokenizer=tokenizer,
    device=device,
    parallel=True,
    num_tokens=128,   # ← Change here
    num_runs=10,      # ← Change here
    batch_size=4,     # ← Change here
)
```

## Dependencies

```bash
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
matplotlib>=3.5.0
numpy>=1.20.0
```

## Notes

- **Memory Usage**: Each model variant requires ~2GB VRAM during training
- **Disk Space**: Each checkpoint is ~200MB; plan for ~2GB per variant
- **Training Time**: ~2-4 hours per variant on a single V100/A100
- **Generation Benchmarks**: Include warmup runs to avoid JIT compilation overhead
- **Reproducibility**: Seeds are not fixed; results may vary slightly between runs

## Citation

If you use this experiment setup or methodology, please cite:

```bibtex
@misc{toeplitz_multihead_optimization,
  title={Optimizing Multi-Head Toeplitz Mixers: Architecture and Inference Trade-offs},
  author={Your Name},
  year={2025}
}
```

