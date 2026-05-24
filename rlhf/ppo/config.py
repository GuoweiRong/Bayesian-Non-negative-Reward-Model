### default shared configs 
from peft import LoraConfig


def get_config(tokenizer):
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    generation_kwargs = {
        "max_new_tokens": 512,
        'min_length': -1, 
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
        "begin_suppress_tokens": [tokenizer.eos_token_id],
    }

    eval_generation_kwargs = {
        "max_new_tokens": 512,
        'min_length': -1, 
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "begin_suppress_tokens": [tokenizer.eos_token_id],
    }

    return lora_config, generation_kwargs, eval_generation_kwargs