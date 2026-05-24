from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import evaluate
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from load_datasets import load_train_eval_dataset
from utils import *

# Add the `./reward_models` path to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reward_models')))
import base_trainer
from reward_trainer import SimpleRewardTrainer, RewardDataCollatorWithPadding


@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=1) 
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "The number of training epochs for the reward model."})
    optim: Optional[str] = field(default="adamw_hf",  metadata={"help": "The optimizer to use."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) 
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # data
    dataset: Optional[str] = field(default='rlhf/bon/step1_obtain_gold_score/unified_sampled_gold_score')
 
    # lora
    use_lora: Optional[bool] = field(default=True)
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=100)
    # model and loss
    base_model: Optional[str] =  field(default="google/gemma-2b-it")
    loss_type: Optional[str] = field(default='bt', metadata={'help': "use 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    weight_ratio: Optional[float] = field(default=0.1, metadata={'help': 'the ratio for label smooth or posreg'})
    freeze_pretrained: Optional[bool] = field(default=False)
    # log
    report_to: Optional[str] = field(default='none', metadata={'help': "use 'none', 'wandb'. "})
    log_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test",)
    save_strategy: Optional[str] = field(default="epoch")
    save_steps: Optional[int] = field(default=1000)
    debug: Optional[bool] = field(default=False, metadata={'help': 'if debug=True, only train with 100 samples'})
    


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]
if script_args.use_lora:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{script_args.max_length}_lora{script_args.lora_r}_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"
else:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{script_args.max_length}_fulltrain_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"

device = Accelerator().local_process_index 

training_args = TrainingArguments(
    output_dir=os.path.join(output_name, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    evaluation_strategy=script_args.evaluation_strategy,
    eval_steps=script_args.eval_steps,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing, 
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
)

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast = False)
tokenizer.max_length = script_args.max_length
if 'Llama' in script_args.base_model:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset, eval_dataset = load_train_eval_dataset(script_args.dataset, tokenizer, size=100 if script_args.debug else None)
print('Training dataset size: {}, validation dataset size: {}'.format(len(train_dataset), len(eval_dataset)))


## load model
if len(script_args.attn_implementation):
    model_params = {
        "attn_implementation": script_args.attn_implementation,
    }
else:
    model_params = {}

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model, num_labels=1, device_map=device, 
    torch_dtype=torch.bfloat16,
    **model_params
)

if script_args.freeze_pretrained:
    # for frozon baseline
    mlp_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, 1024, dtype=torch.bfloat16),  
        nn.ReLU(),
        nn.Linear(1024, 1, dtype=torch.bfloat16)  
    )
    mlp_layer.to(device)
    # Replace the classifier with the MLP
    freeze_trainable_parameters(model)
    model.score = mlp_layer # the score is trainable

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print_trainable_parameters(model)


# Define the trainer parameters
trainer_params = {
    "model": model,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "compute_metrics": compute_metrics,
    "data_collator": RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    'loss_type': script_args.loss_type,
    'weight_ratio': script_args.weight_ratio,
}


if script_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=script_args.lora_target_modules,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    trainer_params["peft_config"] = peft_config

trainer = SimpleRewardTrainer(**trainer_params)
print_trainable_parameters(trainer.model)


print('training start')
trainer.train()
