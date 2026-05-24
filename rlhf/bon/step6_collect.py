import os
import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass, field
from typing import Optional
from utils import create_output_directory, calculate_kl_values


# Dataclass for Argument Parsing
@dataclass
class ScriptArguments:
    proxy_score_path: str = field(default='./step4_choose_best_of_n/gemma-2b-it/grm/proxy_score.csv', metadata={'help': 'Path to the proxy score CSV'})
    gold_score_path: str = field(default='./step5_obtain_bon_gold_score/gemma-2b-it/grm/gold_score.csv', metadata={'help': 'Path to the gold score CSV'})
    output_path: str = field(default='./step6_collect/gemma-2b-it/grm', metadata={'help': 'Path to save the output CSV'})
    
    n_values_start: Optional[int] = field(default=1, metadata={"help": "Starting value of N range to consider."})
    n_values_end: Optional[int] = field(default=406, metadata={"help": "Ending value of N range to consider."})
    kl_min: Optional[float] = field(default=0.0, metadata={"help": "Minimum KL value for filtering."})
    kl_max: Optional[float] = field(default=5.0, metadata={"help": "Maximum KL value for filtering."})

def parse_args() -> ScriptArguments:
    """Parse command-line arguments into ScriptArguments dataclass."""
    parser = argparse.ArgumentParser(description="Process KL values and calculate average gold scores.")
    for field_name, field_def in ScriptArguments.__dataclass_fields__.items():
        parser.add_argument(f"--{field_name}", type=type(field_def.default), default=field_def.default, help=field_def.metadata['help'])
    
    args = parser.parse_args()
    return ScriptArguments(**vars(args))


# Process and Calculate Average Gold Scores
def process_gold_scores(filtered_results, df_gold_score, df_proxy_score):
    """Match and calculate average gold scores for each N in filtered results."""
    best_of_n_score_list = []

    for _, row in filtered_results.iterrows():
        n = row['N']
        print('Processing for N =', n)
        # Filter df_bon by specific N
        df_n = df_proxy_score[df_proxy_score['N'] == n]
        
        # Match and collect gold scores
        matched_gold_scores = []
        for _, bon_row in df_n.iterrows():
            id_value, order_value = bon_row['id'], bon_row['order']
            gold_score = df_gold_score[(df_gold_score['id_ids'] == id_value) & 
                                       (df_gold_score['order_ids'] == order_value)]['gold_rewards'].values
            matched_gold_scores.append(gold_score[0])

        # Calculate and store average score
        avg_gold_score = np.mean(matched_gold_scores) 
        best_of_n_score_list.append(avg_gold_score)

    filtered_results['avg_gold_score'] = best_of_n_score_list
    return filtered_results


# Process and Calculate Average proxy Scores
def processing_proxy_scores(filtered_results, df_proxy_score):
    best_of_n_score_list = []
    n1_scores = df_proxy_score[df_proxy_score['N'] == 1]['rewards']
    baseline_min = np.mean(n1_scores)
    for idx, data in filtered_results.iterrows():
        n, kl = data['N'], data['kl']
        print('Processing for N =', n)
        # Filter df_bon by specific N
        df_n = df_proxy_score[df_proxy_score['N'] == n]
   
        proxy_scores_list = []
        # Iterate over each row in df_n
        for _, row in df_n.iterrows():
            proxy_score = row['rewards'] 
            # Collect all matching gold_score records
            proxy_scores_list.append(proxy_score)

        avg_proxy_score = np.mean(proxy_scores_list)
        avg_proxy_score = avg_proxy_score - baseline_min
        best_of_n_score_list.append(avg_proxy_score)

    filtered_results['avg_proxy_score'] = best_of_n_score_list

    return filtered_results
 

def collect():
    # Parse arguments
    script_args = parse_args()

    # Create output directory
    if not os.path.exists(script_args.output_path):
        os.makedirs(script_args.output_path)

    # Set up N values range and calculate filtered KL values
    N_values = np.arange(script_args.n_values_start, script_args.n_values_end)
    filtered_results = calculate_kl_values(N_values, kl_min=script_args.kl_min, kl_max=script_args.kl_max)

    # Load gold score and BON data
    df_gold_score = pd.read_csv(script_args.gold_score_path)
    df_proxy_score = pd.read_csv(script_args.proxy_score_path)

    # Process results and save
    gold_df  = process_gold_scores(filtered_results.copy(), df_gold_score, df_proxy_score)
    proxy_df = processing_proxy_scores(filtered_results.copy(), df_proxy_score)
    gold_out  = gold_df[['N', 'kl', 'avg_gold_score']].sort_values('N')
    proxy_out = proxy_df[['N', 'kl', 'avg_proxy_score']].sort_values('N')
    gold_out.to_csv(os.path.join(script_args.output_path, 'gold_results.csv'), index=False)
    proxy_out.to_csv(os.path.join(script_args.output_path, 'proxy_results.csv'), index=False)




    # plot_df = process_gold_scores(filtered_results, df_gold_score, df_proxy_score)
    # proxy_df = processing_proxy_scores(plot_df, df_proxy_score)
    # proxy_df.to_csv('%s/proxy_results.csv'%script_args.output_path, index=False)
    # plot_df.to_csv('%s/gold_results.csv'%script_args.output_path, index=False)

if __name__ == "__main__":
    collect()
