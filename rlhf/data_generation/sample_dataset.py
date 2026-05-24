import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from utils import create_output_directory, save_results_in_parquet_splits



@dataclass
class ScriptArguments:
    data_path: Optional[str] = field(default="dataset/Unified-Feedback", metadata={"help": "Path to the data file."})
    category: Optional[str] = field(default="all", metadata={"help": "Data Category"})
    mode: Optional[str] = field(default="train", metadata={"help": "'train', and 'test'"})
    train_size: Optional[int] = field(default=20000, metadata={"help": "train size"})
    test_size: Optional[int] = field(default=1000, metadata={"help": "test size"})
    save_path: Optional[str] = field(default='rlhf/data', metadata={"help": "Directory to save results."})
    save_name: Optional[str] = field(default='unified_sampled', metadata={"help": "Dataset Name."})
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

        

script_args = parse_args()
# Load the dataset
print("Load the dataset.")
ds = load_dataset(script_args.data_path, script_args.category, split=script_args.mode)
# Shuffle the dataset
ds_shuffled = ds.shuffle(seed=42)  # Using a seed for reproducibility
#
print("Sample the training dataset.")
subset_20k = ds_shuffled.select(range(script_args.train_size))

print("Sample the testing dataset.")
subset_1k = ds_shuffled.select(range(script_args.train_size, script_args.train_size+script_args.test_size))

save_path = create_output_directory(script_args.save_path, script_args.save_name)
# Now subset_20k contains 20,000 unique samples and subset_1k contains 1,000 unique samples
save_results_in_parquet_splits(subset_20k, num_splits=script_args.num_splits, save_path=save_path, mode='train')
save_results_in_parquet_splits(subset_1k, num_splits=script_args.num_splits, save_path=save_path, mode='test')
