from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
from utils import get_trainable_weights


@dataclass
class GRMDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    # padding: Union[bool, str, PaddingStrategy] = True
    padding: Union[bool, str, PaddingStrategy] = "max_length"

    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
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
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # pad labels
        paded_length = batch["input_ids"].shape[1]
        label_paded = []
        for feature in features:
            label_chosen_paded = torch.tensor(feature["label_chosen"].tolist() + [self.label_pad_token_id] * (paded_length - len(feature["label_chosen"])) , dtype=torch.int64)
            label_rejected_paded = torch.tensor(feature["label_rejected"].tolist() + [self.label_pad_token_id] * (paded_length - len(feature["label_rejected"])) , dtype=torch.int64)
            label_paded.extend([label_chosen_paded.view(1, -1), label_rejected_paded.view(1, -1)])
        label_paded = torch.concatenate(label_paded, dim=0)
            
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "label": label_paded,  
        }
        return batch



class GRMRewardTrainer(RewardTrainer):    
    def __init__(self, **kwargs):
        self.reference_free = kwargs.pop('reference_free', True)
        self.reference_model = kwargs.pop('reference_model', None)
        self.sft_only = kwargs.pop('sft_only', True)
        self.no_logsigmoid_sft = kwargs.pop('no_logsigmoid_sft', False)
        self.weight_ratio = kwargs.pop('weight_ratio', 0.01)
        self.beta = kwargs.pop('beta', 0.1)
        self.label_pad_token_id = -100
        self.use_lora = kwargs.pop('use_lora', True)
        self.info_to_save = kwargs.pop('info_to_save', {})
        self.KL_ratio = kwargs.pop('KL_ratio',0.00001)
        super(GRMRewardTrainer, self).__init__(**kwargs)


    def get_batch_logps(
        self, 
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens load_train_eval_dataset
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # logits, kl_loss,  rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits, _,  rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if not self.reference_free:
            with torch.no_grad():
                ref_logits = self.reference_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2) # chosen_ids
        kidx = jidx + 1                # rejected_ids  
        reward_loss = -nn.functional.logsigmoid(rewards[jidx] - rewards[kidx]).mean()

        ## text-generation regularization
        if self.weight_ratio > 0:
            logps = self.get_batch_logps(logits, inputs['label'])
            if self.sft_only:
                pi_logratios = logps[jidx] # positive

                if self.no_logsigmoid_sft:
                    dpo_loss = - pi_logratios.mean()
                else:
                    dpo_loss = -F.logsigmoid(self.beta * (pi_logratios)).mean()
            else:
                pi_logratios = logps[jidx] - logps[kidx]  
                if self.reference_free or self.sft_only:
                    ref_logratios = torch.tensor(0.0)
                else:
                    ref_logps = self.get_batch_logps(ref_logits, inputs['label'])
                    ref_logratios = ref_logps[jidx] - ref_logps[kidx]

                pi_logratios = pi_logratios.to(rewards[0].device)
                ref_logratios = ref_logratios.to(rewards[0].device)
                dpo_loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()

            loss = self.weight_ratio * dpo_loss + (1 - self.weight_ratio) * reward_loss
        else:
            loss = reward_loss

        # print(self.KL_ratio)
        # loss += self.KL_ratio*kl_loss.mean()
        if return_outputs:
            return loss, {}
        return loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            logits, _, rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logps = self.get_batch_logps(logits, inputs['label'])
            if self.reference_free:
                dpo_logp_diff = logps
            else:
                ref_logits = self.reference_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
                ref_logps = self.get_batch_logps(ref_logits, inputs['label'])
                dpo_logp_diff = logps - ref_logps

        return (None, dpo_logp_diff.reshape(-1, 2), rewards.reshape(-1, 2))



    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save and self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            ## add config
            model.config.vhead_layer_type = self.info_to_save['layer_type']
            model.config.vhead_num_neurons = self.info_to_save['num_neurons']
            model.config.vhead_num_layers = self.info_to_save['num_layers']
            if self.use_lora:
                state_dict = get_trainable_weights(model.base_model.model)
                print("--- Saving the following weights: ---")
                for key in state_dict.keys():
                    print(key)  
                # for key in state_dict.keys():
                #     print(key)  
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
                # state_dict = get_trainable_weig hts(model)
                state_dict = model.state_dict()
                model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(output_dir)
