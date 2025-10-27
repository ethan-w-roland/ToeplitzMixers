import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging
from itertools import chain

logging.set_verbosity(40)

def prep_dataset(
    num_proc: int,
    tokenizer: AutoTokenizer,
    seq_len: int
) -> Dict[str, Dataset]:
    """    
    Args:
        num_proc: Number of processes for parallel tokenization
        tokenizer: HuggingFace tokenizer
    """
    dset_name = "SimpleStories/SimpleStories"
    train_ds = load_dataset(dset_name, split="train")
    test_ds = load_dataset(dset_name, split="test")

    print("Dataset columns:", train_ds.column_names)
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    
    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Batched tokenization with EOS tokens added."""
        tokenized = tokenizer.batch_encode_plus(
            examples["story"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        # Add EOS token to end of each document
        for token_list in tokenized["input_ids"]:
            token_list.append(tokenizer.eos_token_id)
        return tokenized
    
    def tokenize_and_chunk(dataset, split_name, seq_len):
        """Concatenate all documents and chunk into fixed-size sequences."""
        print(f"\nTokenizing {split_name} dataset with {num_proc} processes...")
        
        # Tokenize in parallel batches (FAST!)
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )
        
        print(f"Concatenating and chunking {split_name} into sequences of {seq_len} tokens...")
        
        # Concatenate all token lists - chain.from_iterable avoids intermediate list creation
        all_tokens = np.fromiter(chain.from_iterable(tokenized["input_ids"]), dtype=np.int32)
        
        total_tokens = len(all_tokens)
        num_complete_chunks = total_tokens // seq_len
        remainder = total_tokens % seq_len
        
        print(f"  Total tokens: {total_tokens:,}, creating {num_complete_chunks:,} chunks...")
        
        # Chunk using numpy reshape for maximum speed
        # Truncate to exact multiple of seq_len, then reshape
        truncated_tokens = all_tokens[:num_complete_chunks * seq_len]
        chunked_array = truncated_tokens.reshape(num_complete_chunks, seq_len)
        
        # Convert back to list of lists for Dataset
        chunks = chunked_array.tolist()
        
        print(f"  Complete chunks: {num_complete_chunks:,}")
        print(f"  Dropped tokens: {remainder}")
        
        return Dataset.from_dict({'input_ids': chunks})
    
    # Pack and chunk both splits
    train_ds = tokenize_and_chunk(train_ds, "train", seq_len=seq_len)
    test_ds  = tokenize_and_chunk(test_ds, "test", seq_len=seq_len)

    return {
        "train": train_ds,
        "test": test_ds,
    }


def write_dataset(
    datasets: Dict[str, Dataset],
    out_dir: Path,
    tokenizer: AutoTokenizer,
    seq_len: int,
) -> None:
    """
    Write datasets to HuggingFace Dataset format and collect metadata.
    
    Each split will be saved to its own directory:
        out_dir/train/
        out_dir/test/
    """
    total_tokens_train = 0
    total_tokens_test = 0
    meta = {}

    for split, dataset in datasets.items():
        print(f"\nSaving {split} dataset...")
        
        # Save dataset in HuggingFace format (creates Arrow/Parquet files)
        dataset_path = out_dir / split
        dataset.save_to_disk(str(dataset_path))
        print(f"  Saved to: {dataset_path}")
        print(f"  Number of examples: {len(dataset)}")

        # ---------- per‑split statistics ----------
        # Count tokens across all chunks
        total_tokens = int(np.sum([len(ids) for ids in dataset["input_ids"]]))
        
        # Show last 100 tokens from last chunk as example
        last_chunk = dataset[-1]["input_ids"]
        example_tokens = last_chunk[-100:] if len(last_chunk) >= 100 else last_chunk
        example_text = tokenizer.decode(
            example_tokens, 
            skip_special_tokens=False
        )

        meta[split] = {
            "total_tokens": total_tokens,
            "num_examples": len(dataset),
            "example": example_text,
            "dataset_path": str(dataset_path),
        }

        if split == "train":
            total_tokens_train += total_tokens
        else:
            total_tokens_test += total_tokens

    # ---------- global statistics ----------
    meta["all"] = {
        "total_tokens_train": total_tokens_train,
        "total_tokens_test": total_tokens_test,
        "vocab_size": len(tokenizer),
        "tokenizer": tokenizer.name_or_path,
        "seq_len": seq_len,
    }

    # ---------------------------------------------------- #
    # dump metadata.json                                   #
    # ---------------------------------------------------- #
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {metadata_path}")


# --------------------------------------------------------------------------- #
# main preparation sequence                                                   #
# --------------------------------------------------------------------------- #

def run(
    out_dir: Path, 
    num_proc: int,
    seq_len: int,
) -> None:
    """
    Main data preparation pipeline using packed tokenization.
    
    Args:
        out_dir: Output directory for datasets
        num_proc: Number of parallel processes for tokenization
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"Vocab size: {len(tokenizer)}")

    datasets = prep_dataset(
        num_proc=num_proc,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

    # Write datasets and metadata
    write_dataset(
        datasets,
        out_dir,
        tokenizer,
        seq_len,
    )

    print(f"\n✓ Done! Packed HuggingFace datasets + metadata.json written to {out_dir}")
    print(f"train_dataset = load_from_disk('{out_dir}/train')")
    print(f"test_dataset = load_from_disk('{out_dir}/test')")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    cur_dir = Path(__file__).parent
    ap = argparse.ArgumentParser(
        description="Prepare SimpleStories dataset with PACKED tokenization (standard for LLM pretraining)"
    )
    ap.add_argument(
        "--out_dir", 
        default=cur_dir / "data", 
        help="Directory to write HuggingFace dataset files"
    )
    ap.add_argument(
        "--num_proc", 
        type=int, 
        default=64,
        help="Number of parallel processes for tokenization"
    )
    ap.add_argument(
        "--seq_len", 
        type=int, 
        default=512,
        help="Sequence length for tokenization"
    )
    args = ap.parse_args()

    run(
        out_dir=Path(args.out_dir),
        num_proc=args.num_proc,
        seq_len=args.seq_len,
    )