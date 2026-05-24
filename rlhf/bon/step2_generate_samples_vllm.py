from transformers import AutoTokenizer
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
from load_datasets import load_data2generateVLLM

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
    tensor_parallel_size:int = field(default=1,metadata={"help":"maxsize GPU number"})
    gpu_memory_utilization : float = field(default=1.00)
    debug: Optional[bool] = field(default=False)

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

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.save_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'



    # Load and process dataset
    dataset = load_data2generateVLLM(script_args.data_path, tokenizer, script_args.debug)
    print('Size of Total Dataset: %s'%(len(dataset)))

    # Inference
    from vllm import LLM, SamplingParams
    llm = LLM(
    model=script_args.model_path,  
    tensor_parallel_size=script_args.tensor_parallel_size,        
    gpu_memory_utilization=script_args.gpu_memory_utilization,   
)
    sampling = SamplingParams(
    max_tokens=script_args.max_new_tokens,  
    temperature=0.7,
    top_p=0.95,
    n=script_args.N,
    seed=42,
    
)
    


    local_results = []
    for i in tqdm(range(0, len(dataset), script_args.batch_size), desc="Generating responses"):
        batch = dataset[i:i + script_args.batch_size]
        # print(f'batch is {batch}')
        prompts = batch['prompt_text']
        # print(f'prompts is {prompts}')
        outputs = llm.generate(prompts, sampling)
        # print(f'outputs is {outputs}')

        B = len(prompts)
        for b in range(B):
            resp = outputs[b]                  
            for cand in resp.outputs:
                local_results.append({
                    "id":     batch["id"][b],
                    "source": batch["source"][b],
                    "input":  batch["input"][b],
                    "output": cand.text,
                })
    
    save_results_in_parquet_splits(local_results, num_splits=script_args.num_splits, save_path=output_dir, mode='test')

# Run main function
if __name__ == "__main__":
    generate_samples()
