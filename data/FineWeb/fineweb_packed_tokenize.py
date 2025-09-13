import torch
import os
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
import pyarrow as pa
import shutil
from dotenv import load_dotenv

load_dotenv()
data_root = os.getenv('DATA_ROOT')

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

all_tokens = torch.tensor([])
def all_packed_tokenization(example):
	n_ctx = context_length # global context_length
	tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False,
		).input_ids

	tokens = torch.flatten(tokens, start_dim=0)
	global all_tokens
	all_tokens = torch.cat((all_tokens, tokens), dim=0)
	if all_tokens.shape[0] > n_ctx:
		batch_size = len(all_tokens) // n_ctx
		length = n_ctx * batch_size
		tokens = all_tokens[:length].reshape(batch_size, n_ctx)
		all_tokens = all_tokens[length:]
		return {'input_ids': tokens.to(torch.long)}
	else:
		return {'input_ids': []}
	

def packed_tokenization(example):
	n_ctx = context_length # global context_length
	tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False,
		).input_ids
	tokens = torch.flatten(tokens, start_dim=0)
	batch_size = len(tokens) // n_ctx
	length = n_ctx * batch_size
	tokens = tokens[:length].reshape(batch_size, n_ctx)
	return {'input_ids': tokens}

def tokenization(example):
	# global padding_side, context_length
	n_ctx = context_length
	tokens = tokenizer.batch_encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			padding='max_length',
			padding_side=padding_side, 
			max_length=n_ctx
		)
	return tokens


def map_dataset(train_path, test_path, split_index=50000, packed=False, fineweb=True, context=512, minimum_length=0):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	if fineweb:
		# fineweb loaders
		train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).skip(split_index)
		test_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).take(split_index)

	else:
		# finemath loaders
		train_text = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=16).skip(split_index)
		test_text = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=16).take(split_index)
	
	if packed:
		batch = False
		tokenize = all_packed_tokenization
	else:
		batch = True
		tokenize = tokenization

	if minimum_length:
		train_text = train_text.filter(lambda x: x['token_count'] > minimum_length)
		test_text = test_text.filter(lambda x: x['token_count'] > minimum_length)
		
	train_dataset = train_text.map(tokenize, batched=batch)
	test_dataset = test_text.map(tokenize, batched=batch)
	train_dataset.save_to_disk(train_path)
	test_dataset.save_to_disk(test_path)
	print ('Datasets saved to disk')
	return

def debatch(example):
	batch_size = len(example['input_ids'])
	keys = list(example.keys())
	for key in keys:
		if key != 'input_ids':
			example.pop(key, None)
	debatched_inputs = [{'input_ids': tokens} for tokens in example["input_ids"][0]]
	if not debatched_inputs: return [{'input_ids': torch}]
	return pa.Table.from_pylist(debatched_inputs)

# user-defined vals
fineweb = False
packed = False
prefix = 'fineweb-edu' if fineweb else 'finemath'
context_length = 512
padding_side = 'right'
pad_contraction = '-lpad' if padding_side == 'left' else ''
train_path = f"{data_root}/{prefix}-tokenized-train-c{context_length}{pad_contraction}-8k"
test_path = f"{data_root}/{prefix}-tokenized-test-c{context_length}{pad_contraction}-8k"

if __name__ == '__main__':
	map_dataset(train_path, test_path, packed=packed, fineweb=fineweb)
	train_dataset = load_from_disk(train_path)
	test_dataset = load_from_disk(test_path)
	if packed:
		train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) > 0)
		test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) > 0 )
		print (train_dataset[0]['input_ids'])
		test_dataset = test_dataset.map(debatch, batched=True, batch_size=1)
		print (test_dataset[0])
		test_dataset.save_to_disk(test_path+'-debatched')
		shutil.rmtree(test_path)
		train_dataset = train_dataset.map(debatch, batched=True, batch_size=1)
		print (train_dataset[0])
		train_dataset.save_to_disk(train_path+'-debatched')
		shutil.rmtree(train_path)
	







