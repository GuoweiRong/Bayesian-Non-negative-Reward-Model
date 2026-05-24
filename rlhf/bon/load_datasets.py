from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def load_train_eval_dataset(data_path, tokenizer, size=None, model_name=''):
    train_dataset = build_dataset_UF(data_path, tokenizer, split='train', size=size, model_name=model_name) 
    eval_dataset = build_dataset_UF(data_path, tokenizer, split='test', model_name=model_name)

    return train_dataset, eval_dataset


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, model_name=''):
    
    ds = load_dataset(data_path, split=split)
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
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
        if 'grm' in model_name:
            # add label mask for sft and dpo training
            prompt = [example['conv_A'][0]]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected,
                # "margin": margin, # GRM does not need this
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin, 
            }
        

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    # ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# Function to load and prepare the dataset
def build_datasets_inference(data_path, tokenizer, split='', size=None, max_length=1024, w_order=False):
    ds = load_dataset(data_path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    
    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": max_length, "return_tensors": "pt"}
        
        if not isinstance(example['output'], str):
            answer = ''
        else:
            answer = example['output']
            
        messages = [{"role": "user", "content": example['input']},
                {"role": "assistant", "content": answer}]
      
        prompt_plus_response = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode_plus(prompt_plus_response, **kwargs)

        if w_order:
            return {
            "input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0],
            "source": example['source'], "id": example['id'], "order": example["order"]}

        else:
            return {
            "input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0],
            "source": example['source'], "id": example['id']}

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    remove_columns = []
    if w_order:
        for name in ds.column_names:
            if 'input_ids' not in name and 'attention' not in name and 'source' not in name and 'id' not in name and 'order' not in name:
                remove_columns.append(name)
    else:
        for name in ds.column_names:
            if 'input_ids' not in name and 'attention' not in name and 'source' not in name and 'id' not in name:
                remove_columns.append(name)
    ds = ds.remove_columns(remove_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds


# Function to load and process dataset
def load_data2generate(data_path, tokenizer, N, debug=False):
    dataset = load_dataset(data_path, split='test')
    if debug == True:
        dataset = dataset.select(range(0,1))
    dataset = dataset.map(lambda x: {
        'id': x['id'],
        'source': x['source'],
        'input': x['conv_A'][0]['content'], # user content
        'prompt': [
            # {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": x['conv_A'][0]['content']}
        ]
    })

    def tokenize_function(examples):
        inputs_ = tokenizer.apply_chat_template(examples["prompt"], tokenize=False, add_generation_prompt=True)
        tokenized_inputs = tokenizer(inputs_, truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'id': examples['id'],
            'source': examples['source'],
            'input': examples['input']
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return concatenate_datasets([tokenized_dataset] * N)


def load_data2generateVLLM(data_path, tokenizer, debug=False):
    dataset = load_dataset(data_path, split='test')
    if debug == True:
        dataset = dataset.select(range(0,1))
    dataset = dataset.map(lambda x: {
        'id': x['id'],
        'source': x['source'],
        'input': x['conv_A'][0]['content'], # user content
        'prompt': [
            # {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": x['conv_A'][0]['content']}
        ]
    })

    def tokenize_function(examples):
        inputs_ = tokenizer.apply_chat_template(examples["prompt"], tokenize=False, add_generation_prompt=True)
        return {
            'prompt_text': inputs_,
            'id': examples['id'],
            'source': examples['source'],
            'input': examples['input']
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


# Custom collate function
def custom_collate(batch):
    return {
        "id": [item["id"] for item in batch],
        "source": [item["source"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch])
    }

def custom_collate_w_order(batch):
    id = [item["id"] for item in batch]
    source = [item["source"] for item in batch]
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    order = [item["order"] for item in batch]

    return {
        "id": id,
        "source": source,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "order": order,
    }


def prepare_data_loader(dataset, tokenizer, batch_size, sampler, collate_fn_type='default'):
    if collate_fn_type == 'default':
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, collate_fn=data_collator)
    elif collate_fn_type == 'custom':
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, collate_fn=custom_collate)
    elif collate_fn_type == 'custom_w_order':
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, collate_fn=custom_collate_w_order)
    return data_loader
