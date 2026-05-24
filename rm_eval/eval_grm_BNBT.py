from dataclasses import dataclass, field
from typing import Optional, Literal
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from load_eval_datasets import load_eval_dataset
from grm_utils_BNBT import load_model_withhead, model_withhead_forward


@dataclass
class ScriptArguments:
    per_device_eval_batch_size: Optional[int] = field(default=8)
    max_length: Optional[int] = field(default=1024) 
    base_model: Optional[str] = field(default="Ray2333/GRM-llama3-8B-sftreg")
    peft_name: Optional[str] = field(default='')
    layer_type: Optional[str] = field(default='mlp') 
    num_layers: Optional[int] = field(default=1)
    log_dir: Optional[str] = field(default='./eval_reward_grm')
    task: Optional[Literal['unified', 'hhh', 'mtbench']] = field(default='unified')
    save_all_data: Optional[bool] = field(default=False)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

accelerator = Accelerator()
device = Accelerator().local_process_index 

model_name = script_args.base_model
log_path = os.path.join(script_args.log_dir, model_name.split('/')[-1], script_args.task)
if accelerator.is_main_process and not os.path.exists(log_path):
    os.makedirs(log_path)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
tokenizer.model_max_length = script_args.max_length
print(tokenizer.padding_side)
if tokenizer.pad_token == None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# load model
model = load_model_withhead(model_name, script_args.peft_name, tokenizer, device, \
                            layer_type=script_args.layer_type, num_layers=script_args.num_layers)

model.eval()
if hasattr(model, "v_head"):
    model.v_head.eval()

# load dataset
eval_dataset = load_eval_dataset(script_args.task, tokenizer)
print('size of test dataset: ', len(eval_dataset))

#### inference
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_data_loader = DataLoader(eval_dataset, batch_size=script_args.per_device_eval_batch_size, drop_last=False, collate_fn=data_collator)
eval_data_loader = accelerator.prepare(eval_data_loader)

full_chosen_prompts = []
full_rejected_prompts = []
full_rewards_chosen = []
full_rewards_rejected = []
full_source_ids = []
pbar = tqdm(total=len(eval_dataset) // script_args.per_device_eval_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(eval_data_loader):
        reward_chosen_tensors = model_withhead_forward(model, batch["input_ids"], batch["attention_mask_chosen"], device, forward_type='reward') 
        reward_rejected_tensors = model_withhead_forward(model, batch["input_ids_rejected"], batch["attention_mask_rejected"], device, forward_type='reward') 
        full_rewards_chosen.extend(reward_chosen_tensors)
        full_rewards_rejected.extend(reward_rejected_tensors)
        full_chosen_prompts.extend(batch['input_ids'])
        full_rejected_prompts.extend(batch['input_ids_rejected'])
        if 'source_id' in batch.keys():
            full_source_ids.extend(batch['source_id'])
        pbar.update(1)

full_chosen_prompts = tokenizer.batch_decode(full_chosen_prompts, skip_special_tokens = True)
full_rejected_prompts = tokenizer.batch_decode(full_rejected_prompts, skip_special_tokens = True)

# print(full_rewards_chosen)
full_rewards_chosen = [x.item() for x in full_rewards_chosen]
full_rewards_rejected = [x.item() for x in full_rewards_rejected]
if 'source_id' in batch.keys():
    full_source_ids = [x.item() for x in full_source_ids]

# local_n = len(full_rewards_chosen)
# print("local_n =", local_n, "rank =", accelerator.process_index)
accelerator.wait_for_everyone()



all_chosen_prompts = accelerator.gather_for_metrics(full_chosen_prompts)
all_rejected_prompts = accelerator.gather_for_metrics(full_rejected_prompts)
all_rewards_chosen = accelerator.gather_for_metrics(full_rewards_chosen)
all_rewards_rejected = accelerator.gather_for_metrics(full_rewards_rejected)

# if accelerator.is_main_process:
#     print("gathered_n =", len(all_rewards_chosen), "expected~", local_n * accelerator.num_processes)
# if accelerator.is_main_process and accelerator.num_processes > 1:
#     assert len(all_rewards_chosen) == local_n * accelerator.num_processes, "gather 没有正确拼接各卡数据"
# if accelerator.is_main_process:
#     print("type(all_chosen_prompts) =", type(all_chosen_prompts))
#     print("type(all_chosen_prompts[0]) =", type(all_chosen_prompts[0]))
#     print("type(all_rewards_chosen[0]) =", type(all_rewards_chosen[0]))



if 'source_id' in batch.keys():
    all_source_ids = accelerator.gather_for_metrics(full_source_ids)


if accelerator.is_main_process:
    evaluation_result = {
        'chosen_prompts': all_chosen_prompts,
        'rejected_prompts': all_rejected_prompts,
        'chosen_rewards': all_rewards_chosen,
        'rejected_rewards': all_rewards_rejected,
        
    }
    if 'source_id' in batch.keys():
        evaluation_result['source_ids'] = all_source_ids
    dataframe = pd.DataFrame(evaluation_result)
    accuracy = (dataframe['chosen_rewards'] > dataframe['rejected_rewards']).mean()
    print('accuracy: ', accuracy)
    if not script_args.save_all_data:
        if dataframe.shape[0] > 1000:
            dataframe = dataframe.head(1000)
    dataframe.to_csv(os.path.join(log_path, 'eval_data.csv'))
    with open(os.path.join(log_path,'accuracy.txt'), 'w+') as f:
        f.write(str(accuracy))
    


