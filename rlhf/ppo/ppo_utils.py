import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import numpy as np
import pandas as pd          
tqdm.pandas()
import matplotlib.pyplot as plt
from typing import Optional  


def plot_curve(script_args, mean_scores, std_scores):
    save_path = os.path.join(script_args.log_dir, script_args.wandb_name, 'scores.png')
    plt.plot(mean_scores)
    plt.fill_between(np.arange(len(mean_scores)), np.array(mean_scores) - np.array(std_scores), np.array(mean_scores) + np.array(std_scores), alpha=0.5)
    plt.savefig(save_path)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def eval_model(ppo_trainer, eval_dataset, tokenizer, accelerator, script_args, name='', eval_generation_kwargs=None):
    full_prompts = []
    full_response_tensors = []
    kl1_list, kl2_list, kl3_list = [], [], []
    full_source_ids, full_id_ids = [], []
    
    eval_data_loader = DataLoader(eval_dataset, batch_size=script_args.eval_batch_size, drop_last=False, collate_fn=collator)
    eval_data_loader = accelerator.prepare(eval_data_loader)

    pbar = tqdm(total=len(eval_dataset) // script_args.eval_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(eval_data_loader):
            query_tensors = batch['input_ids']
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **eval_generation_kwargs) 
            full_response_tensors.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])

            model_inputs = ppo_trainer.prepare_model_inputs(query_tensors, response_tensors)
            all_logprobs, _, _, masks = ppo_trainer.batched_forward_pass(
                ppo_trainer.model,
                query_tensors,
                response_tensors,
                model_inputs,
                return_logits=False,
            )
            with ppo_trainer.optional_peft_ctx():
                ref_logprobs, _, _, _ = ppo_trainer.batched_forward_pass(
                    ppo_trainer.model if ppo_trainer.is_peft_model else ppo_trainer.ref_model,
                    query_tensors,
                    response_tensors,
                    model_inputs,
                    return_logits=False,
                )
            diff = (all_logprobs - ref_logprobs) * masks
            kl1 = diff.sum(dim=-1)
            kl2 = (0.5 * diff.square()).sum(dim=-1)
            kl3 = (diff).abs().sum(dim=-1)

            kl1_list.extend([x.item() for x in kl1])
            kl2_list.extend([x.item() for x in kl2])
            kl3_list.extend([x.item() for x in kl3])

            if 'source' in batch.keys():
                full_source_ids.extend(batch['source'])
            if 'id' in batch.keys():
                full_id_ids.extend(batch['id'])
            pbar.update(1)

    full_prompts = tokenizer.batch_decode(full_prompts)
    full_responses = tokenizer.batch_decode(full_response_tensors)
    if 'source' in batch.keys():
        full_source_ids = [x for x in full_source_ids]
    if 'id' in batch.keys():
        full_id_ids = [x for x in full_id_ids]

    accelerator.wait_for_everyone()
    all_prompts = accelerator.gather_for_metrics(full_prompts)
    all_responses = accelerator.gather_for_metrics(full_responses)
    all_kl1_list = accelerator.gather_for_metrics(kl1_list)
    all_kl2_list = accelerator.gather_for_metrics(kl2_list)
    all_kl3_list = accelerator.gather_for_metrics(kl3_list)
    if 'source' in batch.keys():
        all_source_ids = accelerator.gather_for_metrics(full_source_ids)
    if 'id' in batch.keys():
        all_id_ids = accelerator.gather_for_metrics(full_id_ids)

    if accelerator.is_main_process:
        evaluation_result = {
            'prompts': all_prompts,
            'responses': all_responses,
            'kl1': all_kl1_list,
            'kl2': all_kl2_list,
            'kl3': all_kl3_list,
        }
        if 'source' in batch.keys():
            evaluation_result['source_ids'] = all_source_ids
        if 'id' in batch.keys():
            evaluation_result['id_ids'] = all_id_ids
        dataframe = pd.DataFrame(evaluation_result)
        dataframe.to_csv(os.path.join(script_args.log_dir, script_args.wandb_name,'eval_outputs_{}.csv'.format(name)))



def transfer_template_rm(prompt, response, tokenizer, rm_tokenizer):
    # transfer template from gemma to current tokenizer
    if 'gemma' in tokenizer.name_or_path:
        prompt = prompt.replace('<bos>', '')
        response = response.replace('<eos>', '')

        prompt_lis = prompt.split("<start_of_turn>user\n")[1:]
        messages = []
        for promp in prompt_lis:
            res = promp.split('<start_of_turn>model\n')
            if len(res) == 2 and len(res[1]):
                query, reply = res[0], res[1]
                query, reply = query.replace("<end_of_turn>\n", ''), reply.replace("<end_of_turn>\n", '')
                messages.extend(
                    [{'content': query, 'role': 'user'},
                    {'content': reply, 'role': 'assistant'}]
                )
            else:
                query = res[0] 
                query = query.replace("<end_of_turn>\n", '')
                messages.append(
                    {'content': query, 'role': 'user'},
                )
        prompt_trans = rm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_trans, response
    else:
        raise NotImplementedError


def build_dataset_unified(data_path, tokenizer, script_args, split='', size=None):
    ds = datasets.load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    source_dict = {'argilla/ultrafeedback-binarized-preferences-cleaned':0,
                    'Anthropic/hh-rlhf': 1,
                    'flan_v2_flan2021': 2,
                    'ultrachat': 3,
                    'evol_instruct': 4,
                    'false_qa': 5,
                    'Dahoas/synthetic-instruct-gptj-pairwise': 6,
                    'flan_v2_cot': 7,
                    'flan_v2_p3': 8,
                    'truthful_qa': 9,
                    'lmsys/chatbot_arena_conversations': 10,
                    'openai/summarize_from_feedback(comparisons)': 11,
                    'sharegpt': 12,
                    'flan_v2_niv2': 13,
                    'berkeley-nest/Nectar': 14,
                    'openai/webgpt_comparisons': 15}

    def formatting_func(example):
        kwargs = {"return_tensors": "pt"}
        # kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        messages = example['conv_A'][:-1]
        prompt_plus_response = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode_plus(prompt_plus_response, **kwargs)

        return {
            'query': prompt_plus_response,
            "input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0],
            "source": example['source'], "id": example['id']
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds

