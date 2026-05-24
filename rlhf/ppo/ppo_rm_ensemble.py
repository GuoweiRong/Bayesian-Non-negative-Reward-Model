import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np
import pandas as pd          
tqdm.pandas()
from ppo_utils import print_trainable_parameters, collator, eval_model, build_dataset_unified, transfer_template_rm, plot_curve
from rm_utils import load_reward_model, RMEnsemble
from config import get_config


@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=True, metadata={'help': 'Whether to disable wandb or not.'})
    log_dir: Optional[str] = field(default='./logs_ppo/')
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    eval_batch_size: Optional[int] = field(default=1)
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "loading model in 8 bit or bfloat16"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    init_kl_coef: Optional[float] = field(default=0.0, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},)
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={'help': "use '' if you don't want attention acceleration or meet error here."})
    max_length: Optional[int] = field(default=1024)
    wandb_name: Optional[str] = field(default='ppo_baseline_reward', metadata={"help": "Name for this experiment"})
    dataset_path: Optional[str] = field(default='', metadata={'help': 'training dataset path'})
    eval_dataset_path: Optional[str] = field(default='')
    base_model_name: Optional[str] = field(default='', metadata={'help':"the path to the sft model; need to merge if using lora"})
    reward_base_model: Optional[str] = field(default='')
    reward_peft_path: Optional[str] = field(default='', metadata={'help': 'Separate each path with a comma'})
    ensemble_method: Optional[str] = field(default='avg')
    eval_every: Optional[int] = field(default=6)
    normalize_rewards: Optional[bool] = field(default=True)
    adap_kl_ctrl: Optional[bool] = field(default=False)
    debug: Optional[bool] = field(default=False)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# Remember to use a merged sft model if using lora 
base_model_name = script_args.base_model_name
print('base model: ', base_model_name)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

accelerator = Accelerator()
gpu_id= Accelerator().local_process_index 
set_seed(8888)
print('process: {}'.format(gpu_id))
if accelerator.is_main_process and not os.path.exists(os.path.join(script_args.log_dir, script_args.wandb_name)):
    os.makedirs(os.path.join(script_args.log_dir, script_args.wandb_name))


config = PPOConfig(
    model_name=base_model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    max_grad_norm=5,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    optimize_cuda_cache=True,
    init_kl_coef=script_args.init_kl_coef,
    tracker_project_name='ppo',
    tracker_kwargs={"wandb":{"name":script_args.wandb_name}},
)

# load reward model
reward_peft_model_paths = script_args.reward_peft_path.split(',')
reward_models = RMEnsemble(script_args.ensemble_method, base_model_name=base_model_name, peft_path_list=reward_peft_model_paths)
reward_models.load_reward_models(script_args, gpu_id)

# load tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(reward_peft_model_paths[0], use_fast = False)
train_dataset = build_dataset_unified(script_args.dataset_path, tokenizer, script_args, split='train')
eval_dataset = build_dataset_unified(script_args.eval_dataset_path, tokenizer, script_args, split='test')
if script_args.debug:
    train_dataset = train_dataset.select(range(100))
    eval_dataset = eval_dataset.select(range(40))
print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

# load fixed configs 
lora_config, generation_kwargs, eval_generation_kwargs = get_config(tokenizer)
model_params = {
    "torch_dtype": torch.bfloat16,
    "load_in_8bit": False,
}

if script_args.load_in_8bit:
    model_params["load_in_8bit"] = True
    model_params.pop("torch_dtype")

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_name,
    peft_config=lora_config,
    device_map=gpu_id,
    **model_params
)
print_trainable_parameters(model)
model.pretrained_model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer, dataset=train_dataset, data_collator=collator, optimizer=optimizer
)


print("Training........")
epochs = script_args.epochs
mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    'length_mean': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
}
history_mean_rm, history_std_rm = 0, 1
name = 'epoch_{}_batch_{}'.format(0, 0)
eval_model(ppo_trainer, eval_dataset, tokenizer, accelerator, script_args, name, eval_generation_kwargs)

for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)
    for i, batch in enumerate(ppo_trainer.dataloader):
        print('epoch {}, batch {}'.format(epoch, i))
        query_tensors = batch["input_ids"]

        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs) 

        full_responses = tokenizer.batch_decode(response_tensors)
        lengths = [len(x) for x in full_responses]
        batch['response'] = full_responses
 
        # Compute score
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        if tokenizer.chat_template == reward_models.rm_tokenizers[0].chat_template:
            encoded_prompt_response = [reward_models.rm_tokenizers[0].encode_plus(query + response, **kwargs) for query, response in zip(batch['query'], batch['response'])]
        else:
            # changing template for different reward model and base model
            temp_lis = [(transfer_template_rm(query, response, tokenizer, reward_models.rm_tokenizers[0])) for query, response in zip(batch['query'], batch['response'])]
            encoded_prompt_response = [reward_models.rm_tokenizers[0].encode_plus(query + response, **kwargs) for query, response in temp_lis]
        
        with torch.no_grad():
            reward_tensors = reward_models.forward(encoded_prompt_response)
        rewards = [r.item() for r in reward_tensors]
        
        # normalize using the first batch statistics
        if script_args.normalize_rewards:
            if epoch == 0 and i == 0:
                all_rewards = accelerator.gather_for_metrics(rewards)
                history_mean_rm, history_std_rm = np.mean(all_rewards), np.std(all_rewards)

            reward_tensors = [(x - history_mean_rm) / history_std_rm for x in reward_tensors]
            rewards = [(x - history_mean_rm) / history_std_rm for x in rewards]


        ppo_trainer.config.batch_size = len(query_tensors)
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, rewards)
        policy_kl = [stats["objective/kl"]]

        all_rewards = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)
        all_lengths = accelerator.gather_for_metrics(lengths)
        print("iter {}, batch {}: mean score: {}".format(epoch, i, np.mean(all_rewards)))
        if ppo_trainer.accelerator.is_main_process:
            mean_scores.append(np.mean(all_rewards))
            std_scores.append(np.std(all_rewards))
            plot_curve(script_args, mean_scores, std_scores)

            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['length_mean'].append(np.mean(all_lengths))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append(batch['query'][0] + batch['response'][0])
            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.log_dir, script_args.wandb_name,'data.csv'))
            print("iter {}, batch {}: log finish".format(epoch, i))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)

        # save model
        if i % script_args.eval_every == 0 and i != 0:
            name = 'epoch_{}_batch_{}'.format(epoch, i)
            eval_model(ppo_trainer, eval_dataset, tokenizer, accelerator, script_args, name, eval_generation_kwargs)
            if ppo_trainer.accelerator.is_main_process:
                save_path = os.path.join(script_args.log_dir, script_args.wandb_name, name)
                ppo_trainer.save_pretrained(save_path)
                print("iter {}, batch {}: model saved".format(epoch, i))

# save final model
if i % script_args.eval_every != 0: # not evaluated
    name = 'epoch_{}_batch_{}'.format(epoch, i)
    eval_model(ppo_trainer, eval_dataset, tokenizer, accelerator, script_args, name, eval_generation_kwargs)

    if ppo_trainer.accelerator.is_main_process:
        name = 'epoch_{}_batch_{}'.format(epoch, i)
        save_path = os.path.join(script_args.log_dir, script_args.wandb_name, name)
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))
            