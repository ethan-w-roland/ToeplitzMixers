"""
@author: ethan-w-roland
@date: 2025-07-20
@desc: Toeplitz Mixer Training Script
"""

import argparse, json, os, torch, time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import ToeplitzMixerModel, Config as ToeplitzConfig
from dataloader import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


def load_vocab(token_dir: str) -> int:
    with open(os.path.join(token_dir, "metadata.json")) as f:
        return json.load(f)["all"]["vocab_size"]


def run(
        data_dir: str,
        block_size: int,
        batch_size: int,
        lr: float,
        epochs: int,
) -> None:

    assert torch.cuda.is_available()
    device = "cuda"
    
    # --- Model & Data ---

    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    vocab_size = load_vocab(data_dir)
    config = ToeplitzConfig(vocab_size=vocab_size, block_size=block_size)
    model = ToeplitzMixerModel(config).to(device)
    model = torch.compile(model)

    loader = DataLoader(
        filename=f"{data_dir}/train.bin",
        B=batch_size,
        T=block_size,
        device=device,
        pin_memory=True)

    # --- Optimizer ---

    opt = optim.AdamW(model.parameters(), lr=lr)

    # --- Training Loop ---

    for _ in range(epochs):

        pbar = tqdm(range(len(loader)), ncols=100)

        for _ in pbar:

            # --- Gather Data ---

            data = loader.next_batch()
            x, y = data[:, :-1], data[:, 1:]

            pred = model(x)
            loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)),
                y.reshape(-1),
                reduction="mean",
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_description(f"loss={loss.item():.3f}")


    # --- Generate ---
    data = loader.next_batch()
    data = data[:, :128]
    print(tokenizer.decode(data[0]))
    print('-' * 100)

    print("FAST GEN")
    t1 = time.time()
    pred = model.generate(data, 128)
    t2 = time.time()
    print(tokenizer.decode(pred[0]))
    print(f"Time: {t2 - t1:.2f}s")
    print('-' * 100)

    print("SLOW GEN")
    t1 = time.time()
    pred = model.generate_old(data, 128)
    t2 = time.time()
    print(tokenizer.decode(pred[0]))
    print(f"Time: {t2 - t1:.2f}s")



# --- CLI ---
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="./data")
    ap.add_argument("--batch_size",  type=int, default=256)
    ap.add_argument("--block_size",  type=int, default=256)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--epochs",      type=int, default=1)
    args = ap.parse_args()

    run(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs)