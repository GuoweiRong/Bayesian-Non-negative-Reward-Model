from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import torch
import numpy as np
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import argparse
from load_datasets import build_datasets_inference, prepare_data_loader
from utils import create_output_directory


@dataclass
class ScriptArguments:
    per_device_batch_size: Optional[int] = field(default=64, metadata={"help": "The batch size per device during evaluation."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "The maximum sequence length."})
    data_path: Optional[str] = field(default="./step4_choose_best_of_n/gemma-2b-it/bon_selected_proxy_grm_drop_duplicates", metadata={"help": "Path to the data file."})
    method: Optional[str] = field(default="BNBT", metadata={'help': "use 'BNBT', 'grm', 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    model_path: Optional[str] = field(default="Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback", metadata={"help": "The gold reward model to use."})
    save_path: Optional[str] = field(default='./step5_obtain_bon_gold_score/gemma-2b-it', metadata={"help": "Directory to save results."})


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


# Evaluation function
def evaluate_and_collect_results(model, data_loader, tokenizer, accelerator, batch_size: int) -> Dict[str, List]:
    """Evaluate and return results."""
    full_prompts, full_rewards, full_source_ids, full_id_ids, full_order_ids = [], [], [], [], []
    # pbar = tqdm(total=len(data_loader) * batch_size // accelerator.num_processes)
    pbar = tqdm(total=len(data_loader))
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            reward_tensors = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask"].to(model.device)).logits.reshape(-1)
            full_rewards.extend(reward_tensors)
            full_prompts.extend(batch['input_ids'])
            if 'source' in batch.keys():
                full_source_ids.extend(batch['source'])
            if 'id' in batch.keys():
                full_id_ids.extend(batch['id'])
            if 'order' in batch.keys():
                full_order_ids.extend(batch['order'])
            pbar.update(1)

    full_prompts = [x.rstrip(tokenizer.pad_token) for x in tokenizer.batch_decode(full_prompts)]
    full_rewards = [x.item() for x in full_rewards]
    if 'source' in batch.keys():
        full_source_ids = [x for x in full_source_ids]
    if 'id' in batch.keys():
        full_id_ids = [x for x in full_id_ids]
    if 'order' in batch.keys():
        full_order_ids = [x.item() for x in full_order_ids]

    accelerator.wait_for_everyone()
    # Gather metrics for all processes
    return {
        'prompts': accelerator.gather_for_metrics(full_prompts),
        'gold_rewards': accelerator.gather_for_metrics(full_rewards),
        'source_ids': accelerator.gather_for_metrics(full_source_ids) if full_source_ids else [],
        'id_ids': accelerator.gather_for_metrics(full_id_ids) if full_id_ids else [],
        'order_ids': accelerator.gather_for_metrics(full_order_ids) if full_order_ids else [],
    }


def obtain_bon_gold_score():
    # Parse arguments
    script_args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.method)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=False)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(script_args.model_path, num_labels=1, device_map=accelerator.local_process_index, torch_dtype=torch.bfloat16)
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset and DataLoader
    dataset = build_datasets_inference(script_args.data_path, tokenizer, split='test', max_length=script_args.max_length, w_order=True)
    print('Size of Dataset: %s'%(len(dataset)))
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.local_process_index, shuffle=False)
    data_loader = prepare_data_loader(dataset, tokenizer, script_args.per_device_batch_size, sampler=sampler, collate_fn_type='custom_w_order')
    # data_loader = accelerator.prepare(data_loader)

    # Run evaluation and gather results
    evaluation_result = evaluate_and_collect_results(model, data_loader, tokenizer, accelerator, script_args.per_device_batch_size)
    
    # Save results to CSV
    if accelerator.is_main_process:
        df = pd.DataFrame(evaluation_result)
        df = df.drop_duplicates(subset=["id_ids", "order_ids"])
        df.to_csv(os.path.join(output_dir, 'gold_score.csv'), index=False)


if __name__ == "__main__":
    obtain_bon_gold_score()
