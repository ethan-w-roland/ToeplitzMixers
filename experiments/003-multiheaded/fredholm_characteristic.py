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
from toep_copy import MLPMixer as CopyMixer
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
	plt.tight_layout()
	plt.show()
	plt.close()
	return 

@torch.no_grad()
def plot_all_windings(weight_vectors, n_initials=50000):
	for weight_vector in weight_vectors:
		initial_values = torch.tensor([np.exp(3.141592653589793j * (2*t/n_initials)) for t in range(n_initials)])
		output = toeplitz_symbol(initial_values, weight_vector).numpy()
		real_output = output.real
		imag_output = output.imag
		plt.plot(real_output, imag_output, alpha=1, linewidth=0.15)
	plt.axis('on')
	plt.tight_layout()
	plt.show()
	plt.close()
	return

@torch.no_grad()
def calculate_winding_numbers(weight_vectors, n_initials=50000):
	all_windings = []
	for i, weight_vector in enumerate(weight_vectors):
		initial_values = torch.tensor([np.exp(3.141592653589793j * (2*t/n_initials)) for t in range(n_initials)])
		output = toeplitz_symbol(initial_values, weight_vector).numpy()
		total_arg = 0
		prev_arg = np.angle(output[0], deg=False)
		for point in output[1:]:
			arg = np.angle(point, deg=False)

			# branch point arithmetic (numpy treats the negative real axis as the branch)
			if point.real < 0:
				if arg < 0 and prev_arg > 0:
					rotation = (np.pi + arg) + (np.pi - prev_arg)
				elif arg > 0 and prev_arg < 0:
					rotation = (np.pi - arg) + (np.pi + prev_arg)
				else:
					rotation = arg - prev_arg
			else:
				rotation = arg - prev_arg

			# print (f'Prev Arg: {prev_arg}, Arg: {arg}, Rotation: {rotation}')
			total_arg += rotation
			prev_arg = arg
			# print (f'Total arg: {total_arg}')

		# the following seems to round decently to account for finite differences
		winding_number = int(np.round(total_arg / (2 * np.pi), decimals=0))

		all_windings.append(winding_number)
		print (f"Layer {i} Fredholm Index: {winding_number}")
	return all_windings

@torch.no_grad()
def plot_initial(n_initials=50000):
	initial_values = torch.tensor([np.exp(3.141592653589793j * (2*t/n_initials)) for t in range(n_initials)])
	real_output = initial_values.real
	imag_output = initial_values.imag
	plt.scatter(real_output, imag_output, alpha=1, color='white', linewidth=0.25)
	plt.axis('on')
	plt.tight_layout()
	plt.show()
	plt.close()
	return

def vector_to_matrix(v: torch.Tensor) -> torch.Tensor:
	"""
	Given a vector v of shape (m,), returns an (m x m) matrix M
	where M[i, j] = v[j - i] if j >= i, and 0 otherwise.

	For example, if v = [a, b, c, d] then M will be:

	[ a  b  c  d ]
	[ 0  a  b  c ]
	[ 0  0  a  b ]
	[ 0  0  0  a ]
	"""
	v = v.reshape(-1)  # Ensure v is a 1D tensor
	m = v.shape[0]

	i, j = torch.meshgrid(
		torch.arange(m, device=v.device),
		torch.arange(m, device=v.device),
		indexing="ij",
	)

	M = torch.where(
		j >= i, v[j - i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
	)
	return M

@torch.no_grad()
def plot_windings_grid(weight_vectors, n_initials=50000):
	fig, axes = plt.subplots(4, 4, figsize=(8, 8)) # Create a 4x4 grid of subplots
	axes = axes.flatten()

	for i, ax in enumerate(axes):
		print (i)
		initial_values = torch.tensor([np.exp(3.141592653589793j * (2*t/n_initials)) for t in range(n_initials)])
		output = toeplitz_symbol(initial_values, weight_vectors[i]).numpy()
		real_output = output.real
		imag_output = output.imag
		ax.plot(real_output, imag_output, color='white', alpha=1, linewidth=0.1)
		ax.axis('off')

	plt.tight_layout()
	plt.show()
	plt.close()
	return

def plot_weights(weight_vectors):
	fig, axes = plt.subplots(4, 4, figsize=(8, 8)) # Create a 4x4 grid of subplots
	axes = axes.flatten()

	for i, ax in enumerate(axes):
		ax.imshow(vector_to_matrix(weight_vectors[i]).detach(), cmap='plasma', interpolation='nearest', vmin=-0.015, vmax=0.015)
		# ax.set_title(f'Layer {i}', fontsize='small')
		ax.axis('off')

	plt.tight_layout()
	plt.show()
	plt.close()
	return

def load_clm():
	tokenized_length = 512
	dim = 1024
	layers = 16
	n_heads = None

	model = MLPMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False, copy=False
	)

	checkpoint_path = checkpoint_root + '/fineweb_flat_toep_1024_c512.safetensors'
	load_model(model, checkpoint_path)
	return model

def load_copy_model():
	tokenized_length = 1024
	dim = 512
	layers = 16
	n_heads = None

	model = CopyMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, expanded_convs=False, copy=True
	).float()

	checkpoint_path = checkpoint_root + '/fineweb_copy_flat_toep_512_c1024.json'
	load_model(model, checkpoint_path)
	return model

colormap = plt.cm.viridis
cycler = plt.cycler('color', colormap(np.linspace(0., 1, 16)))
plt.gca().set_prop_cycle(cycler)

if __name__ == "__main__":
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')
	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print("Vocab size: ", n_vocab)
	model = load_clm()

	toeplitz_layers = [model.mixer_blocks[i].token_mixing_layer.weight.squeeze(0) for i in range(len(model.mixer_blocks))]
	weight_vector = toeplitz_layers[5] # 8 has high char
	# weight_vector = torch.zeros(512)
	# weight_vector[1] = 1
	# print (weight_vector)
	plot_winding(weight_vector)
	# plot_all_windings(toeplitz_layers)
	print (calculate_winding_numbers(toeplitz_layers))

	# weight_vector = nn.Parameter(torch.randn(1, 512))
	# nn.init.kaiming_normal_(weight_vector)
	# weight_vector = weight_vector.squeeze(0)

	# plot_weights(toeplitz_layers)
	plot_windings_grid(toeplitz_layers)


	# plot_initial()
	


