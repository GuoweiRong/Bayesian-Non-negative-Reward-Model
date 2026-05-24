import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from accelerate import Accelerator
import evaluate
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
accuracy = evaluate.load('accuracy')


def is_lora_model(model):
    for key in model.state_dict().keys():
        if 'lora' in key:
            return True
    return False

def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'pretrained_model.' in key:
                key = key.replace('pretrained_model.', '')
            save_dict[key] = state_dict[key]
    return save_dict

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


def grm_compute_metrics(eval_pred):
    rewards = eval_pred.label_ids
    reward_accuracy = (rewards[:, 0] > rewards[:, 1]).mean()
    
    predictions = eval_pred.predictions
    accuracy = (predictions[:, 0] > predictions[:, 1]).mean()
    return {
        'dpo_accuracy': accuracy,
        'reward_accuracy': reward_accuracy
    }


def print_trainable_parameters(model, print_trainable_name=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if print_trainable_name:
                print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def freeze_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def create_output_directory(log_dir: str, wandb_name: str):
    output_path = os.path.join(log_dir, wandb_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


# Function to save results as Parquet files
def save_results_in_parquet_splits(results, num_splits, save_path, mode='test'):
    results_df = pd.DataFrame(results)
    dataset_with_results = Dataset.from_pandas(results_df)
    
    split_size = len(dataset_with_results) // num_splits
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(dataset_with_results)
        split = dataset_with_results.select(range(start, end))
        file_path = f"{save_path}/{mode}-0000{i}-of-0000{num_splits}.parquet"
        split.to_parquet(file_path)


# Define the KL equation
def kl_equation(N):
    return np.log(N) - (N - 1) / N


# Calculate and filter KL values
def calculate_kl_values(N_values, kl_min=0, kl_max=5):
    kl_values = [kl_equation(N) for N in N_values]
    results = pd.DataFrame({'N': N_values, 'kl': kl_values})
    return results[(results['kl'] >= kl_min) & (results['kl'] <= kl_max)]


# Define function to get highest rewards within N items per group
def get_highest_within_n(group, n):
    return group.head(n).nlargest(1, 'rewards')