import pandas as pd
import numpy as np
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from utils import save_results_in_parquet_splits, create_output_directory, calculate_kl_values, get_highest_within_n

# Define the dataclass for command-line arguments
@dataclass
class ScriptArguments:
    model_type: Optional[str] = field(default="BNBT", metadata={'help': "use 'BNBT', 'grm', 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    data_path: Optional[str] = field(default="./step3_obtain_proxy_score/gemma-2b-it/grm", metadata={"help": "Path to the data file."})

    n_values_start: Optional[int] = field(default=1, metadata={"help": "Starting value of N range to consider."})
    n_values_end: Optional[int] = field(default=406, metadata={"help": "Ending value of N range to consider."})
    kl_min: Optional[float] = field(default=0.0, metadata={"help": "Minimum KL value for filtering."})
    kl_max: Optional[float] = field(default=5.0, metadata={"help": "Maximum KL value for filtering."})

    save_path: Optional[str] = field(default='./step4_choose_best_of_n/gemma-2b-it', metadata={"help": "Directory to save results."})

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


# Process 'prompts' field into 'input', 'answer', and 'message'
def process_row(data):
    input_text = data.split("<start_of_turn>user\n")[-1].split("<end_of_turn>")[0]
    answer_text = data.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0]
    answer_text = answer_text.replace('<bos>', '').replace('<eos>', '')
    message = [{'content': input_text, 'role': 'user'}, {'content': answer_text, 'role': 'assistant'}]
    return pd.Series([input_text, answer_text, message], index=['input', 'output', 'message'])


# Process and save results for each method
def processing(method, filtered_results, data_df, output_dir):
    # Add order column
    data_df['order'] = data_df.groupby('id_ids').cumcount() + 1
    # List to hold grouped DataFrames
    all_grouped_scores = []

    # Group by and process each N in filtered_results
    for idx, n in enumerate(tqdm(filtered_results['N'].tolist(), desc=f"Processing groups for {method}")):
        grouped_max_scores = data_df.groupby('id_ids').apply(get_highest_within_n, n)
        grouped_max_scores['N'] = n
        all_grouped_scores.append(grouped_max_scores)
       
    # Concatenate all DataFrames
    final_df = pd.concat(all_grouped_scores).reset_index(drop=True)
    # Apply processing to 'prompts' field
    new_columns = final_df['prompts'].apply(process_row)
    final_df = pd.concat([final_df, new_columns], axis=1)
    final_df.rename(columns={'id_ids': 'id', 'source_ids': 'source'}, inplace=True)
    
    # Save results to CSV
    save_path = os.path.join(output_dir, f'bon_selected_proxy_{method}.csv')
    final_df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
    
    # Save deduplicated results
    dedup_df = final_df.drop_duplicates(subset=['id', 'order'])
    dedup_save_path = os.path.join(output_dir, f'bon_selected_proxy_{method}_drop_duplicates')
    os.makedirs(dedup_save_path, exist_ok=True)
    save_results_in_parquet_splits(dedup_df, num_splits=4, save_path=dedup_save_path, mode='test')
    print(f"Saved deduplicated results to {dedup_save_path}")


def choose_best_of_n():
    # Parse arguments
    script_args = parse_args()

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.model_type)

    # Set up N values range and calculate filtered KL values
    N_values = np.arange(script_args.n_values_start, script_args.n_values_end)
    filtered_results = calculate_kl_values(N_values, kl_min=script_args.kl_min, kl_max=script_args.kl_max)
    print('KL_list', filtered_results)

    # Load data
    data = pd.read_csv(f'{script_args.data_path}/proxy_score.csv')

    processing(script_args.model_type, filtered_results, data, output_dir)

if __name__ == "__main__":
    choose_best_of_n()
