import torch
import torch.nn as nn
import os
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel, AutoConfig
from trl import PreTrainedModelWrapper
from peft import PeftModel, PeftConfig
from safetensors import safe_open


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        # get vhead config
        if hasattr(config, "vhead_layer_type"): # config from json first
            self.layer_type = config.vhead_layer_type
        else:
            self.layer_type = kwargs.pop("vhead_layer_type", 'mlp')
        if hasattr(config, 'vhead_num_neurons'):
            num_neurons = config.vhead_num_neurons
        else:
            num_neurons = kwargs.pop("vhead_num_neurons", 1024)
        if hasattr(config, 'vhead_num_layers'):
            num_layers = config.vhead_num_layers
        else:
            num_layers = kwargs.pop("vhead_num_layers", 1)

        if hasattr(config, 'vhead_num_output'):
            num_output = config.vhead_num_output
        else:
            num_output = kwargs.pop("vhead_num_output", 1)

        if self.layer_type == 'linear':
            self.summary = nn.Linear(hidden_size, num_output)
        else:
            module_lis = []
            input_neurons = hidden_size
            for i in range(num_layers):
                module_lis.extend([nn.Linear(input_neurons, num_neurons), nn.ReLU()])
                input_neurons = num_neurons
                
            module_lis.append(nn.Linear(num_neurons, num_output))
            self.summary = nn.Sequential(*module_lis)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if (self.layer_type == 'linear' and output.dtype != self.summary.weight.dtype):
            output = output.to(self.summary.weight.dtype)
        elif (self.layer_type != 'linear' and output.dtype != self.summary[0].weight.dtype):
            output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output)
        return output


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
        "vhead_layer_type",
        'vhead_num_neurons',
        'vhead_num_layers',
        'vhead_num_output',
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if (hasattr(self.v_head.summary, 'weight') and last_hidden_state.device != self.v_head.summary.weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        elif not hasattr(self.v_head.summary, 'weight') and (last_hidden_state.device != self.v_head.summary[0].weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.summary[0].weight.device)
        
        # use the last token value as reward
        last_index = attention_mask.sum(dim=-1) - 1
        value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]
        # value = torch.nn.functional.softplus(value)

        # value [b,10]
        # value = self.v_head(last_hidden_state)[torch.arange(len(last_hidden_state)), last_index]
        # value = torch.relu(value)
        # value = value.sum(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True
    
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F
class ScalarBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1,dtype=torch.float32))

    def forward(self, x):
        # x is just used to match the input/output of Sequential
        return x + F.relu(self.bias)

class BNValueHead(nn.Module):
    """
    Enhanced ValueHead class that implements Weibull reparameterization for reward modeling.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
            # print('*'*50,'hidden_size',hidden_size,'*'*50)
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        
        if hasattr(config, "vhead_layer_type"):
            self.layer_type = config.vhead_layer_type
            
        else:
            self.layer_type = kwargs.pop("vhead_layer_type", 'mlp')
        if hasattr(config, 'vhead_num_neurons'):
            num_neurons = config.vhead_num_neurons
        else:
            num_neurons = kwargs.pop("vhead_num_neurons", 1024)
            # print('*'*50,'elsenum_neurons',num_neurons,'*'*50)
        if hasattr(config, 'vhead_num_layers'):
            num_layers = config.vhead_num_layers
        else:
            num_layers = kwargs.pop("vhead_num_layers", 1)
        if hasattr(config, 'vhead_num_output'):
            num_output = config.vhead_num_output
        else:
            num_output = kwargs.pop("vhead_num_output", 1)
        
       
        # if hasattr(config, 'weibull_dim'):
        #     weibull_dim = config.weibull_dim
        # else:
        #     weibull_dim = kwargs.pop("weibull_dim", 2048)

        
        if self.layer_type == 'linear':
           
            self.linear_l = nn.Linear(hidden_size, num_neurons)
            self.linear_k = nn.Linear(hidden_size, 1)
            self.linear_w = nn.Linear(num_neurons, 1)
            self.linear_kw = nn.Linear(1, 1)
            # self.value_bias = nn.Parameter(torch.zeros(1, dtype=self.linear_l.weight.dtype))
            self.value_bias = ScalarBias()
            # self.summary = nn.Linear(hidden_size, num_output)
            # self.summary = nn.Sequential(
            #     self.linear_l,
            #     self.linear_k,
            #     self.linear_w,
            #     self.linear_kw,
            #     self.value_bias
            # )
        else:
            
            # linear_l: hidden_size2048 -> 1024
            print('*'*10,'num_neurons',num_neurons,'*'*10)
            self.linear_l = self._build_mlp(hidden_size, num_neurons, num_neurons, num_layers,dtype=torch.float32)
            
            # linear_k: hidden_size -> 1
            self.linear_k = self._build_mlp(hidden_size, num_neurons, num_neurons, num_layers,dtype=torch.float32)

            # linear_w: 1024 -> 1
            self.linear_w = nn.Linear(num_neurons, 1,bias=False,dtype=torch.float32)
            self.linear_kw = self._build_mlp(1, 1, num_neurons, num_layers,dtype=torch.float32)
            self.value_bias = ScalarBias()
        
    def _build_mlp(self, input_size, output_size, hidden_size, num_layers,use_bias: bool = True,dtype: torch.dtype = torch.float32):
        
        module_list = []
        module_list.extend([nn.Linear(input_size, hidden_size,bias=use_bias,dtype=dtype), nn.ReLU()])
        # 中间层
        for _ in range(num_layers-1):
            module_list.extend([nn.Linear(hidden_size, hidden_size,bias=use_bias,dtype=dtype), nn.ReLU()])

        module_list.append(nn.Linear(hidden_size, output_size,bias=use_bias,dtype=dtype))
        
        return nn.Sequential(*module_list)

    def reparameterize(self, lbd, kappa):
        def log_max(input, SMALL=1e-10):
            device = input.device
            input_ = torch.max(input, torch.tensor([SMALL]).to(device))
            return torch.log(input_)

        if self.training:
            u = torch.rand_like(lbd)
            z = lbd * (- log_max(1 - u)).pow(1 / kappa)
        else:
            z = lbd * torch.exp(torch.lgamma(1 + 1/kappa))
        return z



    
    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):

        def log_max(input, SMALL=1e-10):
            device = input.device
            input_ = torch.max(input, torch.tensor([SMALL]).to(device))
            return torch.log(input_)
        
        eulergamma = torch.tensor(0.5772, dtype=torch.float32, requires_grad=False).to(Wei_scale.device)
        with torch.autocast(device_type="cuda", enabled=False):
            part1 = Gam_shape * log_max(Wei_scale) - eulergamma * Gam_shape * Wei_shape_res + log_max(Wei_shape_res)
            part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
            part3 = eulergamma + 1 + Gam_shape * log_max(Gam_scale) - torch.lgamma(Gam_shape)
            KL = part1 + part2 + part3
        return -KL.sum(1).mean()




    def forward(self, hidden_states,attention_mask=None,return_theta: bool = False):
        """
        
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        with torch.cuda.amp.autocast(enabled=False):
            output = self.dropout(hidden_states).float()
            for layer in [self.linear_l, self.linear_k, self.linear_w, self.linear_kw, self.value_bias]:
                layer.to(torch.float32)
            if self.layer_type == 'linear':
                if output.dtype != self.linear_l.weight.dtype:
                    output = output.to(self.linear_l.weight.dtype)
            else:
                if output.dtype != self.linear_l[0].weight.dtype:
                    output = output.to(self.linear_l[0].weight.dtype)

            
            z_out = F.relu(self.linear_l(output)).clamp(min=1e-6, max=1e4).float()
            k  = F.softplus(self.linear_k(output)) .clamp(min=1e-6, max=1e4).float()
            # k  = torch.exp(self.linear_k(output)) .clamp(min=1e-6, max=1e4).float()
            k+=1.0
            weibull_lambda = z_out / torch.exp(torch.lgamma(1.0 + 1.0 / k)).float()
            # weibull_lambda = z_out
            pre_out = self.reparameterize(weibull_lambda, k)
            last_index = attention_mask.sum(dim=-1) - 1      # [B]
            theta_last = pre_out[torch.arange(len(pre_out)), last_index, :]
            # pre_out = z_out

            
            if self.layer_type == 'linear':
                z_out_w = F.relu(self.linear_w.weight.transpose(1, 0)) # [hidden_size, 1]
            else:
                z_out_w = F.relu(self.linear_w.weight.T.contiguous()).float()
                

           
    
            k_w = F.softplus(self.linear_kw(z_out_w)).clamp(min=1e-6, max=1e4).float()
            # k_w = torch.exp(self.linear_kw(z_out_w)).clamp(min=1e-6, max=1e4).float()
            k_w +=1.0
            weibull_lambda_w = z_out_w / torch.exp(torch.lgamma(1.0 + 1.0 / k_w)).float()
            # weibull_lambda_w = z_out_w 
            pre_out_w =  self.reparameterize(weibull_lambda_w, k_w)  # [h,1]
            self.last_pre_out_w=pre_out_w
            # pre_out_w = z_out_w
    

            # last_index = attention_mask.sum(dim=-1) - 1  # [batch_size]

            if self.layer_type == 'linear':
                value = self.value_bias(torch.matmul(pre_out, pre_out_w)).squeeze(-1)[torch.arange(len(hidden_states)), last_index]
                # value =  self.value_bias(torch.matmul(pre_out, pre_out_w)).sum(dim=-1,keepdim=True).squeeze(-1)[torch.arange(len(hidden_states)), last_index]
            else:
                value = self.value_bias(torch.matmul(pre_out, pre_out_w)).squeeze(-1)[torch.arange(len(hidden_states)), last_index]
                # value =  self.value_bias(torch.matmul(pre_out, pre_out_w)).sum(dim=-1,keepdim=True).squeeze(-1)[torch.arange(len(hidden_states)), last_index]
            # scores = torch.matmul(pre_out, pre_out_w).mean(dim=-1,keepdim=True)
            # b, t = 0, 0
            # print("scores[b,t,:]:", scores[b, t, :].detach().cpu().tolist())
            # import pdb
            # pdb.set_trace()
            device = hidden_states.device
            gamma_shape = torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=device)
            gamma_scale = torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=device)
 
            kl_z = self.KL_GamWei(gamma_shape, gamma_scale, 1/k, weibull_lambda)
            kl_w = self.KL_GamWei(gamma_shape, gamma_scale, 1/k_w, weibull_lambda_w)
            kl_loss = kl_z + kl_w
       
    
            
            
            # print(f'***************z_out is {z_out}')
            # print(f'***************gamma_shape is {gamma_shape}')
            # print(f'***************gamma_scale is {gamma_scale}')
            # print(f'***************k is {k}')
            # print(f'***************weibull_lambda is {weibull_lambda}')
            # print(f'***************value is {value}')
            # print(f'***************kl_z is {kl_z}')
            # print(f'***************kl_w is {kl_w}')
            # print(f'***************kl_loss is {kl_loss}')
            # import pdb
            # pdb.set_trace()
            if return_theta:
                return value, kl_loss, theta_last.detach()
            return value,kl_loss
            # return value


class AutoModelForCausalLMWithBNValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
        "vhead_layer_type",
        'vhead_num_neurons',
        'vhead_num_layers',
        'vhead_num_output',
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = BNValueHead(self.pretrained_model.config, **v_head_kwargs)
        self.v_head.to(torch.float32)
        
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_theta: bool = True,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss
        # print(f'base_model_output:{base_model_output}')
        # print(f'last_hidden_state[-1]:{last_hidden_state}')

        if (hasattr(self.v_head.linear_l, 'weight') and last_hidden_state.device != self.v_head.linear_l.weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.linear_l.weight.device)
        elif not hasattr(self.v_head.linear_l, 'weight') and (last_hidden_state.device != self.v_head.linear_l[0].weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.linear_l[0].weight.device)


        
        # # use the last token value as reward
        last_index = attention_mask.sum(dim=-1) - 1
        # with torch.cuda.amp.autocast(enabled=False):
        if return_theta:
            value,kl_loss,theta_last=self.v_head(last_hidden_state, attention_mask=attention_mask,return_theta=return_theta)
        else:
            value, kl_loss = self.v_head(last_hidden_state, attention_mask=attention_mask,return_theta=return_theta)
        # value = self.v_head(last_hidden_state, attention_mask=attention_mask)
    
        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        if return_theta:
            return lm_logits, kl_loss, value, theta_last.detach()
        return (lm_logits, kl_loss, value)
        # return (lm_logits,loss, value)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        # print("--- Saving the following weights: ---")
        # for key in v_head_state_dict.keys():
        #         print(key)  # 输出每个权重的名称
        
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True
    
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class



















class GRewardModel(PreTrainedModel):
    config_class = AutoConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        model = AutoModelForCausalLM.from_config(config)
        self.model = model.model
        self.v_head = ValueHead(self.model.config)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        base_model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.hidden_states[-1]

        if hasattr(self.v_head.summary, "weight") and last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        elif not hasattr(self.v_head.summary, "weight") and (
            last_hidden_state.device != self.v_head.summary[0].weight.device
        ):
            last_hidden_state = last_hidden_state.to(self.v_head.summary[0].weight.device)

        # use the last token value as reward
        if torch.any(attention_mask[:, 0] == 0):
            # left padding
            last_index = attention_mask.shape[-1] - 1
        else:
            # right padding
            last_index = attention_mask.sum(dim=-1) - 1
        value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]
        return value

class GRewardModelBN(PreTrainedModel):
    config_class = AutoConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        model = AutoModelForCausalLM.from_config(config)
        self.model = model.model
        self.v_head = BNValueHead(self.model.config)
        self.v_head.to(torch.float32)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        base_model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.hidden_states[-1]
        if (hasattr(self.v_head.linear_l, 'weight') and last_hidden_state.device != self.v_head.linear_l.weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.linear_l.weight.device)
        elif not hasattr(self.v_head.linear_l, 'weight') and (last_hidden_state.device != self.v_head.linear_l[0].weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.linear_l[0].weight.device)

        # use the last token value as reward
        if torch.any(attention_mask[:, 0] == 0):
            # left padding
            last_index = attention_mask.shape[-1] - 1
        else:
            # right padding
            last_index = attention_mask.sum(dim=-1) - 1
        with torch.cuda.amp.autocast(enabled=False):
            value,_ = self.v_head(last_hidden_state, attention_mask=attention_mask)
        return _,_,value

def load_model_withhead(model_name, peft_name, tokenizer, device, \
        layer_type='mlp', num_neurons=1024, num_layers=1, load_in_8bit=False):

    model_config = {
        'device_map': device,
        'vhead_layer_type': layer_type,
        'vhead_num_neurons': num_neurons,
        'vhead_num_layers': num_layers,
    }
    if load_in_8bit:
        model_config['load_in_8bit'] = True
    else:
        model_config['torch_dtype'] = torch.bfloat16

    if 'Mistral' not in model_name:
        model_config['attn_implementation'] = "flash_attention_2"
    
    if not len(peft_name):
        model_config.pop('attn_implementation')
        model = GRewardModelBN.from_pretrained(model_name, **model_config)
        model.model.resize_token_embeddings(len(tokenizer))
        print('*************Loading GRewardModelBN*******************')
    else:
        model = AutoModelForCausalLMWitBNhValueHead.from_pretrained(model_name, **model_config)
        model.pretrained_model.resize_token_embeddings(len(tokenizer))
        assert os.path.exists(peft_name), f"Checkpoint path {peft_name} does not exist!"
        print('*************rm_eval_BNBT_utils********************************************************')
        print('*************Loading AutoModelForCausalLMWithBNValueHead*******************')
    
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(peft_name) and os.path.exists(peft_name):
        peft_config = PeftConfig.from_pretrained(peft_name)
        model = PeftModel(model, peft_config)
        loaded_state_dict = {}

        safetensor_files = sorted([f for f in os.listdir(peft_name) if f.endswith('.safetensors')])
        if len(safetensor_files):
            for safetensor_file in safetensor_files:
                safetensor_path = os.path.join(peft_name, safetensor_file)
                if os.path.exists(safetensor_path):
                    with safe_open(safetensor_path, framework="pt", device=device) as f:
                        # vhead_keys = [k for k in f.keys() if "v_head" in k]
                        for k in f.keys():
                            loaded_state_dict[k] = f.get_tensor(k)
                    # print("v_head keys:", vhead_keys)
                    # for k in vhead_keys:
                    #     v = loaded_state_dict[k]
                    #     print(f"\n==== {k} ====")
                    #     print("shape:", v.shape)
                    #     print(v)
        else:
            loaded_state_dict = torch.load(os.path.join(peft_name, "pytorch_model.bin"))
        missing, unexpected = model.base_model.model.pretrained_model.load_state_dict(loaded_state_dict, strict=False)
        missing, unexpected = model.base_model.model.load_state_dict(loaded_state_dict, strict=False)
    
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    return model

def model_withhead_forward(model, input_ids, attention_mask, device, forward_type='reward', labels=None):
    if isinstance(model, GRewardModel):
        reward_tensors = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    elif forward_type == 'reward':
        _, _, reward_tensors = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    elif forward_type == 'dpo':
        res = model(input_ids.to(device), attention_mask=attention_mask.to(device))
        if len(res) == 3:
            logits, _, _ = res 
        else:
            logits = res.logits
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)
    else:
        raise NotImplementedError
    return reward_tensors