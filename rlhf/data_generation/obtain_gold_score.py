import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional
from utils import create_output_directory, save_results_in_parquet_splits
from load_datasets import build_dataset_UF4gold_score, prepare_data_loader, load_dataset_within_maxlength
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)



@dataclass
class ScriptArguments:
    per_device_batch_size: Optional[int] = field(default=4, metadata={"help": "The batch size per device during evaluation."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "The maximum sequence length."})
    data_path: Optional[str] = field(default="rlhf/data/unified_sampled", metadata={"help": "Path to the data file."})
    model_path: Optional[str] = field(default="Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback", metadata={"help": "The gold reward model to use."})
    save_path: Optional[str] = field(default='rlhf/data', metadata={"help": "Directory to save results."})
    save_name: Optional[str] = field(default="unified_sampled_gold_score", metadata={"help": "Saved file name."})
    mode: Optional[str] = field(default="train", metadata={"help": "'train', and 'test'"})
    num_splits: int = field(default=1, metadata={"help": "Number of splits for saving results"})
    debug: Optional[bool] = field(default=False)
    

def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Set parameters for model training & evaluation.")
    for field_name, field_def in ScriptArguments.__dataclass_fields__.items():
        parser.add_argument(
            f"--{field_name}",
            type=type(field_def.default),
            default=field_def.default,
            help=field_def.metadata.get("help", "")
        )
    args = parser.parse_args()
    return ScriptArguments(**vars(args))


# Main function
def obtain_gold_score(script_args):

    # Initialize Accelerator
    accelerator = Accelerator()
    device = Accelerator().local_process_index 
    print('Curent Device', device)
    print('Number of processes:', accelerator.num_processes)

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.save_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=False)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(script_args.model_path, num_labels=1, device_map=device, torch_dtype=torch.bfloat16)
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare dataset and DataLoader
    dataset = build_dataset_UF4gold_score(script_args.data_path, tokenizer, split=script_args.mode, max_length=script_args.max_length)
    
    if script_args.debug:
        dataset = dataset.select(range(0,100))
    print('Size of %s Dataset: %s'%(script_args.mode, len(dataset)))
        
    # Shard the dataset among processes
    # 分布式数据采样器
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.local_process_index, shuffle=False)
    data_loader = prepare_data_loader(dataset, tokenizer, script_args.per_device_batch_size, sampler=sampler)
    data_loader = accelerator.prepare(data_loader)

    print('Start Inference.')
    # Generate and collect results
    full_chosen_prompts = []
    full_rejected_prompts = []
    full_rewards_chosen = []
    full_rewards_rejected = []
    full_unique_ids = []
    if accelerator.is_main_process:
        pbar = tqdm(total=len(dataset) // script_args.per_device_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            reward_chosen_tensors = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask_chosen"].to(device)).logits.reshape(-1)
            reward_rejected_tensors = model(batch["input_ids_rejected"].to(device), attention_mask=batch["attention_mask_rejected"].to(device)).logits.reshape(-1)
            full_rewards_chosen.extend(reward_chosen_tensors)
            full_rewards_rejected.extend(reward_rejected_tensors)
            full_chosen_prompts.extend(batch['input_ids'])
            full_rejected_prompts.extend(batch['input_ids_rejected'])
            if 'unique_id' in batch.keys():
                full_unique_ids.extend(batch['unique_id'])
            if accelerator.is_main_process:
                pbar.update(1)
    
    full_chosen_prompts = [x.rstrip(tokenizer.pad_token) for x in tokenizer.batch_decode(full_chosen_prompts)]
    full_rejected_prompts = [x.rstrip(tokenizer.pad_token) for x in tokenizer.batch_decode(full_rejected_prompts)]

    full_rewards_chosen = [x.item() for x in full_rewards_chosen]
    full_rewards_rejected = [x.item() for x in full_rewards_rejected]
    if 'unique_id' in batch.keys():
        full_unique_ids = [x.item() for x in full_unique_ids]

    # print(f'Process {accelerator.local_process_index} processed {len(full_chosen_prompts)} prompts')
    # 让所有 GPU/进程在这一点停下来，等待彼此执行完前面的操作后再继续往下走。
    accelerator.wait_for_everyone()
  
    all_chosen_prompts = accelerator.gather_for_metrics(full_chosen_prompts)
    all_rejected_prompts = accelerator.gather_for_metrics(full_rejected_prompts)
    all_rewards_chosen = accelerator.gather_for_metrics(full_rewards_chosen)
    all_rewards_rejected = accelerator.gather_for_metrics(full_rewards_rejected)
    if 'unique_id' in batch.keys():
        all_unique_ids = accelerator.gather_for_metrics(full_unique_ids)
    
    if accelerator.is_main_process:
        evaluation_result = {
            'prompts_A': all_chosen_prompts,
            'prompts_B': all_rejected_prompts,
            'rewards_A': all_rewards_chosen,
            'rewards_B': all_rewards_rejected,
        }
        if 'unique_id' in batch.keys():
            evaluation_result['unique_ids'] = all_unique_ids

        gold_scores_dataframe = pd.DataFrame(evaluation_result)
        # Sort the DataFrame by 'unique_id'
        df_sorted_gold_scores_dataframe = gold_scores_dataframe.sort_values(by='unique_ids')
        df_sorted_gold_scores_dataframe = df_sorted_gold_scores_dataframe.drop_duplicates(subset='unique_ids', keep='first') # remove duplicated sample if there is any
        df_sorted_gold_scores_dataframe = df_sorted_gold_scores_dataframe.reset_index(drop=True)
        df_sorted_gold_scores_dataframe.to_csv(os.path.join(output_dir, 'gold_score_%s.csv'%script_args.mode))

        # replace with the gold scores
        def replace_with_gold_reward(example):
            matching_row = gold_scores_dataframe[gold_scores_dataframe['unique_ids'] == example['unique_id']]
            example['conv_A_rating'] = matching_row.iloc[0]['rewards_A']
            example['conv_B_rating'] = matching_row.iloc[0]['rewards_B']
            return example
        
        # Apply the replacement function to the dataset
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=False)
        dataset_prepared = load_dataset_within_maxlength(script_args.data_path, tokenizer, split=script_args.mode, max_length=script_args.max_length)
        if script_args.debug:
            dataset_prepared = dataset_prepared.select(range(0,100))
        # print('len(dataset_prepared)', len(dataset_prepared))
        assert len(dataset_prepared) == len(df_sorted_gold_scores_dataframe)
        dataset_gold_score = dataset_prepared.map(replace_with_gold_reward)
        dataset_gold_score = dataset_gold_score.remove_columns(['unique_id'])
        save_results_in_parquet_splits(dataset_gold_score, num_splits=script_args.num_splits, save_path=output_dir, mode=script_args.mode)

        

if __name__ == "__main__":
    script_args = parse_args()
    obtain_gold_score(script_args)