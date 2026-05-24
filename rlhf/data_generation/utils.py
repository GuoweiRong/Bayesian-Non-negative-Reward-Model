import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import evaluate
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn


def create_output_directory(log_dir: str, wandb_name: str):
    output_path = os.path.join(log_dir, wandb_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    return output_path

# Function to save results as Parquet files
def save_results_in_parquet_splits(results, num_splits, save_path, mode='test'):
    results_df = pd.DataFrame(results)
    # 将 DataFrame 转换为 Hugging Face 的 datasets.Dataset 对象。
    dataset_with_results = Dataset.from_pandas(results_df)
    
    split_size = len(dataset_with_results) // num_splits
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(dataset_with_results)
        split = dataset_with_results.select(range(start, end))
        file_path = f"{save_path}/{mode}-0000{i}-of-0000{num_splits}.parquet"
        split.to_parquet(file_path)
