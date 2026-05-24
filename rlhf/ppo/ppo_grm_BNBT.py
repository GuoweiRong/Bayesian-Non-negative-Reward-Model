import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, set_seed
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import numpy as np
import pandas as pd          
tqdm.pandas()
from ppo_utils import print_trainable_parameters, collator, eval_model, build_dataset_unified, transfer_template_rm, plot_curve
from rm_utils import load_reward_model
from config import get_config
from model_utils import model_withhead_forward

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[bool] = field(default=False, metadata={'help': 'Whether to disable wandb or not.'})
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
    reward_peft_path: Optional[str] = field(default='')
    gold_reward_model: Optional[str] = field(default='', metadata={'help': '黄金奖励模型路径，用于对比评估'})
    eval_every: Optional[int] = field(default=6)
    layer_type: Optional[str] = field(default='mlp', metadata={'help': 'GRM value head layer type: mlp or linear'})
    num_layers: Optional[int] = field(default=1, metadata={'help': 'GRM value head number of layers'})
    normalize_rewards: Optional[bool] = field(default=True)
    adap_kl_ctrl: Optional[bool] = field(default=True)
    debug: Optional[bool] = field(default=False)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# Remember to use a merged sft model if using lora 
base_model_name = script_args.base_model_name
tokenizer_name = script_args.base_model_name
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
    tracker_kwargs={"wandb":{
        "name": script_args.wandb_name,
        "mode": "offline",
        "dir": os.path.join(script_args.log_dir, script_args.wandb_name)
    }},
)

# load reward model (代理模型)
reward_model, rm_tokenizer, rm_gpu_id = load_reward_model(script_args, gpu_id, rm_type='bnbt')
# load gold reward model (黄金模型，用于对比评估)
gold_reward_model = None
gold_rm_tokenizer = None
gold_rm_gpu_id = None

if script_args.gold_reward_model and script_args.gold_reward_model.strip():
    print(f"加载黄金奖励模型: {script_args.gold_reward_model}")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as AutoTokenizer_Gold
        
        # 简单加载黄金模型（假设是全参数模型）
        gold_reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.gold_reward_model, 
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        gold_rm_tokenizer = AutoTokenizer_Gold.from_pretrained(script_args.gold_reward_model)
        gold_rm_gpu_id = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("黄金奖励模型加载成功")
    except Exception as e:
        print(f"黄金奖励模型加载失败: {e}")
        print("将跳过黄金模型评估")
        gold_reward_model = None

# load tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
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
    'training_samples': [],  # 累计训练样本数 (X轴)
    'proxy_score': [],       # 每个batch的代理奖励分数 (Y轴)
    'proxy_score_normalized': [],  # 标准化后的代理分数 (减去初始值)
    'gold_score': [],        # 每个batch的黄金奖励分数 (绝对分数，无需标准化)
    'kl_mean': [],
    'length_mean': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
}
total_training_samples = 0  # 累计样本计数器
baseline_proxy_score = None  # 基线分数 (第一个batch)
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
        if tokenizer.chat_template == rm_tokenizer.chat_template:
            encoded_prompt_response = [rm_tokenizer.encode_plus(query + response, **kwargs) for query, response in zip(batch['query'], batch['response'])]
        else:
            # changing template for different reward model and base model
            temp_lis = [(transfer_template_rm(query, response, tokenizer, rm_tokenizer)) for query, response in zip(batch['query'], batch['response'])]
            encoded_prompt_response = [rm_tokenizer.encode_plus(query + response, **kwargs) for query, response in temp_lis]
        
        # 代理模型打分
        with torch.no_grad():
            reward_tensors = [model_withhead_forward(reward_model, x['input_ids'], x["attention_mask"], device=rm_gpu_id) for x in encoded_prompt_response] 
        rewards = [r.item() for r in reward_tensors]
        
        # 黄金模型打分 (如果可用)
        gold_rewards = None
        if gold_reward_model is not None:
            try:
                with torch.no_grad():
                    # 为黄金模型准备输入（可能需要不同的tokenizer）
                    if gold_rm_tokenizer.chat_template == rm_tokenizer.chat_template:
                        gold_encoded = encoded_prompt_response
                    else:
                        # 重新编码（适配黄金模型的tokenizer）
                        kwargs_gold = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
                        if tokenizer.chat_template == gold_rm_tokenizer.chat_template:
                            gold_encoded = [gold_rm_tokenizer.encode_plus(query + response, **kwargs_gold) for query, response in zip(batch['query'], batch['response'])]
                        else:
                            temp_lis_gold = [(transfer_template_rm(query, response, tokenizer, gold_rm_tokenizer)) for query, response in zip(batch['query'], batch['response'])]
                            gold_encoded = [gold_rm_tokenizer.encode_plus(query + response, **kwargs_gold) for query, response in temp_lis_gold]
                    
                    gold_reward_tensors = [gold_reward_model(x['input_ids'].to(gold_rm_gpu_id)).logits[0] for x in gold_encoded]
                    gold_rewards = [r.item() for r in gold_reward_tensors]
            except Exception as e:
                print(f"黄金模型推理失败: {e}")
                gold_rewards = None
        
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

            # 更新累计训练样本数
            total_training_samples += len(query_tensors) * accelerator.num_processes
            current_proxy_score = np.mean(all_rewards)
            
            # 设置基线分数 (第一个batch的分数)
            if baseline_proxy_score is None:
                baseline_proxy_score = current_proxy_score
                print(f"设置基线代理分数: {baseline_proxy_score:.4f}")
            
            # 计算标准化分数 (减去基线)
            normalized_proxy_score = current_proxy_score - baseline_proxy_score
            
            # 处理黄金模型分数 (保持绝对分数，无需标准化)
            current_gold_score = None
            if gold_rewards is not None:
                all_gold_rewards = accelerator.gather_for_metrics(gold_rewards)
                current_gold_score = np.mean(all_gold_rewards)
            
            save_data['training_samples'].append(total_training_samples)
            save_data['proxy_score'].append(current_proxy_score)  # 原始代理奖励分数
            save_data['proxy_score_normalized'].append(normalized_proxy_score)  # 标准化代理分数
            save_data['gold_score'].append(current_gold_score if current_gold_score is not None else float('nan'))  # 黄金分数 (绝对值)
            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['length_mean'].append(np.mean(all_lengths))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append([
    query + response for query, response in zip(batch['query'], batch['response'])
])
            
            # 打印分数对比
            if current_gold_score is not None:
                print(f"Batch {i}: 代理分数={current_proxy_score:.4f}(标准化:{normalized_proxy_score:.4f}), "
                      f"黄金分数={current_gold_score:.4f}(绝对值)")
            else:
                print(f"Batch {i}: 代理分数={current_proxy_score:.4f}, 标准化分数={normalized_proxy_score:.4f}")
            
            print(f"🔄 Process {gpu_id}: 开始保存数据...")
            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.log_dir, script_args.wandb_name,'data.csv'))
            print(f"✅ Process {gpu_id}: 数据保存完成")
            print("iter {}, batch {}: log finish".format(epoch, i))

        print(f"🔄 Process {gpu_id}: 等待其他进程同步...")
        # wait for the main process
        accelerator.wait_for_everyone()
        print(f"✅ Process {gpu_id}: 同步完成，继续下一个batch...")
        pbar.update(1)

        # save model
        if i % script_args.eval_every == 0 and i != 0:
            name = 'epoch_{}_batch_{}'.format(epoch, i)
            eval_model(ppo_trainer, eval_dataset, tokenizer, accelerator, script_args, name, eval_generation_kwargs)
            # if ppo_trainer.accelerator.is_main_process:
            #     save_path = os.path.join(script_args.log_dir, script_args.wandb_name, name)
            #     ppo_trainer.save_pretrained(save_path)
            #     print("iter {}, batch {}: model saved".format(epoch, i))

# save final model
if i % script_args.eval_every != 0: # not evaluated
    name = 'epoch_{}_batch_{}'.format(epoch, i)
    eval_model(ppo_trainer, eval_dataset, tokenizer, accelerator, script_args, name, eval_generation_kwargs)

    if ppo_trainer.accelerator.is_main_process:
        name = 'epoch_{}_batch_{}'.format(epoch, i)
        save_path = os.path.join(script_args.log_dir, script_args.wandb_name, name)
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))

# 训练完成后提取绘图数据
if accelerator.is_main_process:
    try:
        # 读取完整的训练数据
        data_csv_path = os.path.join(script_args.log_dir, script_args.wandb_name, 'data.csv')
        if os.path.exists(data_csv_path):
            df = pd.read_csv(data_csv_path)
            
            # 提取Origin绘图需要的核心数据
            plot_data = {
                'training_samples': df['training_samples'].tolist(),
                'kl_mean':df['kl_mean'].tolist(),
                'proxy_score_normalized': df['proxy_score_normalized'].tolist(),
            }
            
            # 如果有黄金模型数据，也提取 (黄金模型使用绝对分数)
            if 'gold_score' in df.columns and not df['gold_score'].isna().all():
                plot_data['gold_score'] = df['gold_score'].tolist()
                print("提取了代理模型标准化分数和黄金模型绝对分数数据")
            else:
                print("只提取了代理模型的标准化分数数据")
            
            # 保存为Origin绘图专用的CSV文件
            plot_df = pd.DataFrame(plot_data)
            ppo_csv_path = os.path.join(script_args.log_dir, script_args.wandb_name, 'ppo_plot_data.csv')
            plot_df.to_csv(ppo_csv_path, index=False)
            
            print(f"✅ Origin绘图数据已保存至: {ppo_csv_path}")
            print(f"数据点数量: {len(plot_df)}")
            print("列包含:", list(plot_data.keys()))
            
            # 显示数据预览
            print("\n📊 数据预览:")
            print(plot_df.head())
            
        else:
            print("❌ 未找到训练数据文件，跳过绘图数据提取")
            
    except Exception as e:
        print(f"❌ 提取绘图数据时出错: {e}")
        
print("🎉 PPO训练完成!")
            