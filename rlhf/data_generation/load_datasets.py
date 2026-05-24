from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def build_dataset_UF4gold_score(data_path, tokenizer, split='', size=None, max_length=1024):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    source_dict = {'argilla/ultrafeedback-binarized-preferences-cleaned':0,
                    'Anthropic/hh-rlhf': 1,
                    'flan_v2_flan2021': 2,
                    'ultrachat': 3,
                    'evol_instruct': 4,
                    'false_qa': 5,
                    'Dahoas/synthetic-instruct-gptj-pairwise': 6,
                    'flan_v2_cot': 7,
                    'flan_v2_p3': 8,
                    'truthful_qa': 9,
                    'lmsys/chatbot_arena_conversations': 10,
                    'openai/summarize_from_feedback(comparisons)': 11,
                    'sharegpt': 12,
                    'flan_v2_niv2': 13,
                    'berkeley-nest/Nectar': 14,
                    'openai/webgpt_comparisons': 15}

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": max_length, "return_tensors": "pt"}
        example['source_id'] = source_dict[example['source']]
        
        chosen_messages = example['conv_A']
        rejected_messages = example['conv_B']

        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        return {
            "input_ids": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    remove_columns = []
    for name in ds.column_names:
        if 'input_ids' not in name and 'attention' not in name and 'source_id' not in name:
            remove_columns.append(name)
    ds = ds.remove_columns(remove_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length and len(x["input_ids_rejected"]) <= max_length, num_proc=30)
    # source_ids = list(range(len(ds)))  # Create unique identifiers
    ds = ds.map(lambda x, idx: {**x, "unique_id": idx}, with_indices=True)
    ds.set_format(type="torch")
    return ds


def load_dataset_within_maxlength(data_path, tokenizer, split='', size=None, max_length=1024):

    # Load the dataset
    ds = load_dataset(data_path, split=split)

    # Optionally limit the size
    if size is not None:
        ds = ds.select(range(0, size))

    # Define a function to calculate lengths for filtering
    def calculate_length(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": max_length}
        chosen_messages = example['conv_A']
        rejected_messages = example['conv_B']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)["input_ids"]
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)["input_ids"]
        return {
            "len_chosen": len(tokens_chosen),
            "len_rejected": len(tokens_rejected),
        }

    # Apply the length calculation to the dataset
    length_ds = ds.map(calculate_length, batched=False, num_proc=30)

    # Filter based on lengths
    valid_indices = [i for i, row in enumerate(length_ds) if row["len_chosen"] <= max_length and row["len_rejected"] <= max_length]

    # Filter the original dataset using the valid indices
    filtered_ds = ds.select(valid_indices)
    filtered_ds = filtered_ds.map(lambda x, idx: {**x, "unique_id": idx}, with_indices=True)

    return filtered_ds

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