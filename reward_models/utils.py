from accelerate import Accelerator
import evaluate
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
accuracy = evaluate.load('accuracy')


def is_lora_model(model):
    for key in model.state_dict().keys():
        if 'lora' in key:
            return True
    return False

def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'pretrained_model.' in key:
                key = key.replace('pretrained_model.', '')
            save_dict[key] = state_dict[key]
    return save_dict

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def BNBT_compute_metrics(eval_pred):
    predictions = eval_pred.label_ids
    # print("[BNBTcompute_metrics] preds type/shape:", type(predictions), getattr(predictions, "shape", None))
    bt_accuracy = (predictions[:, 0] > predictions[:, 1]).mean()
    return {
        'BNBT_accuracy': bt_accuracy
    }

def grm_compute_metrics(eval_pred):
    rewards = eval_pred.label_ids
    # print(rewards)
    reward_accuracy = (rewards[:, 0] > rewards[:, 1]).mean()
    predictions = eval_pred.predictions
    accuracy = (predictions[:, 0] > predictions[:, 1]).mean()
    return {
        'dpo_accuracy': accuracy,
        'reward_accuracy': reward_accuracy
    }


def print_trainable_parameters(model, print_trainable_name=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if print_trainable_name:
                print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def freeze_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def unreeze_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = True
