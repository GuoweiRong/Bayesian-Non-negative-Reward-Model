import numpy as np
import os
import gzip
import json
import random
import torch
import torch.nn as nn
from collections import defaultdict
from datasets import load_dataset, Dataset


# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)
    
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": tokenizer.max_length,
            "return_tensors": "pt"
        }
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = example['chosen'][:-1]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode='', model_name='', Label_noise=False, noise_ratio=0.5):
    _LOCAL_NOISE_SEED = int(os.environ.get("UF_NOISE_SEED", 42))
    _rng = random.Random(_LOCAL_NOISE_SEED)

    try:
        ds = load_dataset(data_path, split=split)
    except:
        ds = load_dataset(data_path, split=split)
    
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode):
        if mode == '1k' or mode == '1K':
            ds = ds.select(range(0, len(ds), 800))
        elif mode == '5k' or mode == '5K':
            ds = ds.select(range(0, len(ds), 160))
        elif mode == '10k' or mode == '10K':
            ds = ds.select(range(0, len(ds), 80))
        elif mode == '20k' or mode == '20K':
            ds = ds.select(range(0, len(ds), 40))
        elif mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20))
        elif mode == '80k' or mode == '80K':
            ds = ds.select(range(0, len(ds), 10))
        elif mode == '160k' or mode == '160K':
            ds = ds.select(range(0, len(ds), 5))
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2))
        elif mode == 'all':
            ds = ds

    if size is not None:
        ds = ds.select(range(0, size))
    
    def formatting_func(example):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": tokenizer.max_length,
            "return_tensors": "pt"
        }
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = [example['conv_A'][0]]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected,
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin,
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10)

    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    # *********************Label Noise**************************
    if Label_noise is True:
        print('*' * 50, f"Loading Noise is {Label_noise}", '*' * 50)
        NOISE_RATIO = noise_ratio
        print('*' * 50, f"Loading NOISE_RATIO is {noise_ratio}", '*' * 50)
        n = len(ds)
        k = int(n * NOISE_RATIO)
        if k > 0:
            flip_indices = set(_rng.sample(range(n), k))

            def flip_pair(example, idx):
                if idx not in flip_indices:
                    return example
                out = dict(example)
                out['input_ids_chosen'], out['input_ids_rejected'] = example['input_ids_rejected'], example['input_ids_chosen']
                out['attention_mask_chosen'], out['attention_mask_rejected'] = example['attention_mask_rejected'], example['attention_mask_chosen']
                if 'label_chosen' in example and 'label_rejected' in example:
                    out['label_chosen'], out['label_rejected'] = example['label_rejected'], example['label_chosen']
                if 'margin' in example:
                    out['margin'] = -example['margin']
                return out

            ds = ds.map(flip_pair, with_indices=True, num_proc=1)
            print("************************Label Noise********************************")

    ds.set_format(type="torch")
    return ds


# for Skywork Reward Preference 80K
def build_dataset_SK(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": tokenizer.max_length,
            "return_tensors": "pt"
        }
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            prompt_template = tokenizer.apply_chat_template(
                [{"content": prompt, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True
            )
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100

            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    ds.set_format(type="torch")
    return ds





def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', model_name='', Label_noise=False, noise_ratio=0.25, help_preference=True):
    if 'Unified' in data_path:
        train_dataset = build_dataset_UF(
            data_path, tokenizer, split='train', size=size, mode=mode,
            model_name=model_name, Label_noise=Label_noise, noise_ratio=noise_ratio
        )
        eval_dataset = build_dataset_UF(
            data_path, tokenizer, split='validation',
            model_name=model_name, Label_noise=False, noise_ratio=noise_ratio
        )
    elif 'Skywork' in data_path:
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.005)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    else:
        dataset = build_dataset(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']

    return train_dataset, eval_dataset