from swift.llm import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from BNRM_ppo_utils import load_model_withhead,load_model_withhead_BNRM
from transformers import AutoTokenizer
from swift.llm import Model, ModelGroup, ModelMeta, register_model
from transformers import AutoConfig


def get_bnbt_rm_model_tokenizer(model_dir, model_info, model_kwargs, load_model, **kwargs):
    device    = model_kwargs.get("device_map", "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(model_info, "config", None) is None:
        model_info.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = load_model_withhead_BNRM(
        model_name=model_dir,
        tokenizer=tokenizer,
        device=device,
        layer_type=getattr(model_info.config, "vhead_layer_type", "mlp"),
        num_neurons=getattr(model_info.config, "vhead_num_neurons", 1024),
        num_layers=getattr(model_info.config, "vhead_num_layers", 1),
    )
    return model, tokenizer


register_model(
    ModelMeta(
        model_type='BNRM',
        model_groups=[],
        template='llama3_1',
        get_function=get_bnbt_rm_model_tokenizer,
    )
)


def get_grm_rm_model_tokenizer(model_dir, model_info, model_kwargs, load_model, **kwargs):
    device    = model_kwargs.get("device_map", "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(model_info, "config", None) is None:
        model_info.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = load_model_withhead(
        model_name=model_dir,
        tokenizer=tokenizer,
        device=device,
        layer_type=getattr(model_info.config, "vhead_layer_type", "mlp"),
        num_neurons=getattr(model_info.config, "vhead_num_neurons", 1024),
        num_layers=getattr(model_info.config, "vhead_num_layers", 1),
    )
    return model, tokenizer


register_model(
    ModelMeta(
        model_type='GRM',
        model_groups=[],
        template='llama3_1',
        get_function=get_grm_rm_model_tokenizer,
    )
)
