import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import HfArgumentParser, AutoModelForSequenceClassification, AutoTokenizer,set_seed
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import numpy as np
import pandas as pd          
tqdm.pandas()
from peft import LoraConfig, PeftModel
import matplotlib.pyplot as plt
from model_utils import load_model_withhead,load_model_withhead_BNBT



def load_reward_model(script_args, gpu_id, rm_type=None):
    ### here we use device map to put large reward models to empty gpus to avoid memory error
    if '7B' in script_args.reward_peft_path:
        rm_gpu_id = {
            0: 4,
            1: 4,
            2: 5,
            3: 5,
        }[gpu_id]
    else:
        rm_gpu_id = gpu_id

    rm_load_params = {
        "num_labels": 1,
        "device_map": rm_gpu_id,
        "torch_dtype": torch.bfloat16,
    }

    if len(script_args.attn_implementation):
        rm_load_params["attn_implementation"] = script_args.attn_implementation

    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_base_model, use_fast = False)
    rm_tokenizer.model_max_length = script_args.max_length
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    if rm_type == 'grm':
        reward_model = load_model_withhead(script_args.reward_base_model, script_args.reward_peft_path, \
                         rm_tokenizer, rm_gpu_id, layer_type=script_args.layer_type, num_layers=script_args.num_layers)
    elif rm_type == 'bnbt':
        reward_model = load_model_withhead_BNBT(script_args.reward_base_model, script_args.reward_peft_path, \
                         rm_tokenizer, rm_gpu_id, layer_type=script_args.layer_type, num_layers=script_args.num_layers)
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_base_model,
            **rm_load_params
            )

        if os.path.exists(script_args.reward_peft_path):
            reward_model = PeftModel.from_pretrained(reward_model, script_args.reward_peft_path)
        if hasattr(reward_model, 'merge_and_unload'):
            reward_model = reward_model.merge_and_unload()
        reward_model.config.pad_token_id = rm_tokenizer.pad_token_id

    return reward_model, rm_tokenizer, rm_gpu_id




class RMEnsemble():
    def __init__(self, ensemble_method='avg', base_model_name='', peft_path_list=[]):
        self.ensemble_method = ensemble_method
        self.base_model_name = base_model_name
        self.peft_path_list = peft_path_list
        self.reward_models = []
        self.gpu_ids = []
        self.rm_tokenizers = []

    def load_reward_models(self, script_args, gpu_id):
        for _ in range(len(self.peft_path_list)):
            reward_model, rm_tokenizer, rm_gpu_id = load_reward_model(script_args, gpu_id)
            self.reward_models.append(reward_model)
            self.gpu_ids.append(rm_gpu_id)
            self.rm_tokenizers.append(rm_tokenizer)

        
    def forward(self, encoded_prompt_response):
        results = []
        with torch.no_grad():
            for i in range(len(self.peft_path_list)):
                reward_tensors = [self.reward_models[i](x['input_ids'].to(self.gpu_ids[i])).logits[0] for x in encoded_prompt_response] 
                results.append(torch.concat(reward_tensors).view(-1, 1))

        if self.ensemble_method == 'avg':
            reward_tensors = torch.concat(results, dim=-1).mean(dim=-1)
        elif self.ensemble_method == 'min':
            reward_tensors = torch.concat(results, dim=-1).min(dim=-1)[0]
        else:
            raise NotImplementedError
        return reward_tensors
    