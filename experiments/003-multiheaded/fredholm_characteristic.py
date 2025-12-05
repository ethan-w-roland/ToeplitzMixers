import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model
import mlflow
import os
from dotenv import load_dotenv
import shutil
import numpy as np
import matplotlib.pyplot as plt 
from toep_mixer_multiheaded import MLPMixer
plt.style.use('dark_background')

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def toeplitz_symbol(z, weight_vector):
	"""
	Toeplitz symbol computation
	args
		z: torch.tensor[C]
		weight vector: 
	"""
	total = torch.zeros(z.shape, dtype=torch.cfloat)
	for i in range(len(weight_vector)):
		total += weight_vector[i] * z ** (-i)
	return total

@torch.no_grad()
def plot_winding(weight_vector, n_initials=50000):
	initial_values = torch.tensor([np.exp(3.141592653589793j * (2*t/n_initials)) for t in range(n_initials)])
	output = toeplitz_symbol(initial_values, weight_vector).numpy()
	real_output = output.real
	imag_output = output.imag
	plt.plot(real_output, imag_output, alpha=1, color='white', linewidth=0.15)
	plt.axis('on')
	plt.show()
	plt.close()


if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)

    tokenized_length = 512
    dim = 1024
    layers = 16
    n_heads = None

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False, copy=False
    )

    checkpoint_path = checkpoint_root + '/fineweb_flat_toep_1024_c512.safetensors'
    load_model(model, checkpoint_path)
    toeplitz_layers = [model.mixer_blocks[i].token_mixing_layer.weight.squeeze(0) for i in range(len(model.mixer_blocks))]
    weight_vector = toeplitz_layers[10]

    # weight_vector = nn.Parameter(torch.randn(1, 512))
    # nn.init.kaiming_normal_(weight_vector)
    # weight_vector = weight_vector.squeeze(0)

    print (weight_vector)
    plot_winding(weight_vector)

