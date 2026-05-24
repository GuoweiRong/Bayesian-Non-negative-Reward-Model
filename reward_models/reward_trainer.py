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
        # print(batch[0])
        # import pdb
        # pdb.set_trace()
        return batch


class SimpleRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop('loss_type', 'bt')
        self.weight_ratio = kwargs.pop('weight_ratio', 0.01)
        self.use_lora = kwargs.pop('use_lora', True)
        super(SimpleRewardTrainer, self).__init__(**kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards  = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        # print('rewards',rewards)
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        # print('rewards_j',rewards_j)
        # print('reward_k',rewards_k)

        if self.loss_type == 'bt':
            # print('rewards_j - rewards_k',rewards_j - rewards_k)
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        elif self.loss_type == 'bt_l2':
            l2 = 0.5 * (rewards_j ** 2 + rewards_k ** 2)
            pair_loss = -nn.functional.logsigmoid(rewards_j - rewards_k) + 0.001 * l2
            loss = pair_loss.mean()
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
    