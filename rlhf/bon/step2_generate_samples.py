from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import torch
import time
from tqdm.auto import tqdm
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
from math import ceil

from utils import create_output_directory, save_results_in_parquet_splits
from load_datasets import load_data2generate

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ScriptArguments:
    batch_size: int = field(default=128, metadata={"help": "Batch size for inference"})
    max_new_tokens: int = field(default=1024, metadata={"help": "Maximum number of new tokens to generate"})
    N: int = field(default=405, metadata={"help": "Number of dataset duplications"})
    data_path: str = field(default='rlhf/data/unified_sampled_gold_score', metadata={"help": "Path to the dataset"})
    model_path: str = field(default='google/gemma-2b-it', metadata={"help": "Path to the policy model checkpoint"})
    save_path: Optional[str] = field(default='./step2_generate_samples', metadata={"help": "Directory to save results."})
    save_name: Optional[str] = field(default='generated_samples_unified', metadata={"help": "Saved file name."})
    num_splits: int = field(default=6, metadata={"help": "Number of splits for saving results"})
    debug: Optional[bool] = field(default=True)

def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Script for generating responses using a Hugging Face model with distributed acceleration.")
    for field_name, field_def in ScriptArguments.__dataclass_fields__.items():
        parser.add_argument(
            f"--{field_name}",
            type=type(field_def.default),
            default=field_def.default,
            help=field_def.metadata.get("help", "")
        )
    args = parser.parse_args()
    return ScriptArguments(**vars(args))



def generate_samples():
    # Parse arguments
    script_args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = Accelerator().local_process_index 

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.save_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(script_args.model_path, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = accelerator.prepare(model)


    # Load and process dataset
    dataset = load_data2generate(script_args.data_path, tokenizer, script_args.N, script_args.debug)
    print('Size of Total Dataset: %s'%(len(dataset)))
   
    # Prepare dataset with accelerator
    total_size = len(dataset)
    chunk_size = ceil(total_size / accelerator.num_processes)
    start_idx = device * chunk_size
    end_idx = min(start_idx + chunk_size, total_size)
    local_dataset = dataset.select(range(start_idx, end_idx))
    print('Size of Local Dataset: %s'%len(local_dataset))
    print('Model & Datasets Loaded.')

    # Inference
    local_results = []
    model.eval()
    for i in tqdm(range(0, len(local_dataset), script_args.batch_size), desc="Generating responses"):
        batch = local_dataset[i:i + script_args.batch_size]
        prompts = {k: torch.tensor(v, device=model.device, dtype=torch.bfloat16 if k == 'attention_mask' else None)
                   for k, v in batch.items()
                   if k in ['input_ids', 'attention_mask']}

        with torch.no_grad():
            outputs = model.module.generate(
                **prompts,
                max_new_tokens=script_args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, 
                top_k=0.0,
                temperature=0.7,
                top_p=0.95
            )
        
        # Remove prompt from generated tokens
        outputs = [tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts["input_ids"], outputs)]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for idx, output in enumerate(decoded_outputs):
            
            local_results.append({
                'id': batch['id'][idx], 
                'source': batch['source'][idx],
                'input': batch['input'][idx],
                'output': output 
            })

    print('Generate %s from device %s'%(len(local_results), device))

    # Synchronize across GPUs after the loop
    accelerator.wait_for_everyone()
    # Gather results from all processes
    all_results = accelerator.gather_for_metrics(local_results)

    # Save results
    if accelerator.is_main_process:
        save_results_in_parquet_splits(all_results, num_splits=script_args.num_splits, save_path=output_dir, mode='test')

# Run main function
if __name__ == "__main__":
    generate_samples()
