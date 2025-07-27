# ToeplitzMixers

Experimental repo for Toeplitz-parameterized Mixer Models for causal language modeling. Contains data preparation, training, and model architecture code. Also implements a simple caching solution for fast autoregressive inference, enabling ~3x speedups over naive autoregressive generation. Dataset used for training is [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories), a modern successor to TinyStories. Training was done on a single H100, taking about 3 minutes to train for the entirety of the SimpleStories dataset.

Install via uv with `uv sync`

TODO: 
- The dataset is currently organized such that each story (of variable length) is placed one after the other in the memory mapped file, separated by [EOS] tokens. The dataloader yields fixed-sized chunks, which means that most training examples will contain elements from more than one story. We may see some accuracy gains by implementing something akin to attention masking, which would prevent the mixer from "attending to" tokens that are before a proceeding [EOS] token.\
- Get further speedups by using Triton / Mojo to accelerate autoregressive inference (and maybe also training). Need to study more on the specifics of this.