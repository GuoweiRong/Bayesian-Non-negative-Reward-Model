from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
import os
from utils import get_trainable_weights

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if 'margin' in feature.keys():
                margins.append(feature['margin'])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "margin": margins,
        }
        return batch


class SimpleRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop('loss_type', 'bt')
        self.weight_ratio = kwargs.pop('weight_ratio', 0.01)
        self.info_to_save = kwargs.pop('info_to_save', {})
        self.use_lora = kwargs.pop('use_lora', True)
        # self.KL_ratio = 0.00001
        self.KL_ratio = kwargs.pop('KL_ratio', 0.00001)
        super(SimpleRewardTrainer, self).__init__(**kwargs)

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits, kl_loss,  rewards  = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        if self.loss_type == 'bt':
            T=1
            loss = - nn.functional.logsigmoid((rewards_j - rewards_k)/T).mean()
            loss += self.KL_ratio*kl_loss.mean()
        elif self.loss_type == 'pos_reg':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() - self.weight_ratio * nn.functional.logsigmoid(rewards_j.mean())
        elif self.loss_type == 'margin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()
        elif self.loss_type == 'labelsmooth':
            loss = - (1-self.weight_ratio) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - self.weight_ratio * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        else:
            raise NotImplementedError

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
    #KL_LOSS
    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save and self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            ## add config
            model.config.vhead_layer_type = self.info_to_save['layer_type']
            model.config.vhead_num_neurons = self.info_to_save['num_neurons']
            model.config.vhead_num_layers = self.info_to_save['num_layers']
            # print(self.use_lora)
            # import pdb
            # pdb.set_trace()
            if self.use_lora:
                state_dict = get_trainable_weights(model.base_model.model)
                print("--- Saving the following weights: ---")
                for key in state_dict.keys():
                    print(key)  
                # import pdb
                # pdb.set_trace()
                model.base_model.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
                model.peft_config['default'].base_model_name_or_path = self.info_to_save['base_model']
                model.peft_config['default'].save_pretrained(output_dir)
                import shutil
                src = os.path.join(output_dir, "model.safetensors")        
                dst = os.path.join(output_dir, "adapter_model.safetensors")
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
            else: # for full training and deepspeed
                # state_dict = get_trainable_weights(model)
                state_dict = model.state_dict()
                model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(output_dir)
    # KLLOSS
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            _, _,  rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return (None, torch.zeros_like(rewards).reshape(-1,2), rewards.reshape(-1,2))
