import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


def attn_forward_hook(self, *args, **kwargs):
    """
    Replace the original attention forward with this to enable prefix
    """

    def _expand_bsz(x, bsz):
        x = x.reshape(x.size(0), self.num_heads, -1).transpose(0,1) # (num_prefix, hidden) -> (num_head, num_prefix, hidden/num_head)
        x = x.unsqueeze(0).expand(bsz, *x.shape) # -> (bsz, num_head, num_prefix, hidden/num_head)
        return x
    
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]
    bsz = hidden_states.size(0)

    if 'past_key_value' not in kwargs or kwargs['past_key_value'] is None:
        if self.reparam:
            prefix_keys = self.prefix_mlp_keys(self.prefix_input_embeds)
            prefix_values = self.prefix_mlp_values(self.prefix_input_embeds)
        else:
            prefix_keys, prefix_values = self.prefix_keys, self.prefix_values
        kwargs['past_key_value'] = (_expand_bsz(prefix_keys, bsz), _expand_bsz(prefix_values, bsz))
    
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']  
            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1], self.num_prefix), dtype=am.dtype, device=am.device), am], dim=-1)
        elif len(args) > 1: # attention mask is passed via positional argument
            am = args[1]
            am = torch.cat([-torch.zeros((*am.shape[:-1], self.num_prefix), dtype=am.dtype, device=am.device), am], dim=-1)
            args = (args[0], am) + args[2:]

    return self.original_forward(*args, **kwargs)


def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    """
    Replace the original "prepare_inputs_for_generation" with this to pass prefix correctly
    """
    original_input_len = input_ids.size(-1)
    if past_key_values:
        input_ids = input_ids[:, -1:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    if past_key_values is not None:
        # Check if we should add extra to attention mask
        if past_key_values[0][0].size(2) != attention_mask.size(1) - 1:
            num_prefix = past_key_values[0][0].size(2) - (attention_mask.size(1) - 1)
            attention_mask = torch.cat([torch.ones((attention_mask.size(0), num_prefix), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=-1)

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


class PrefixTuning:

    def __init__(self, model, num_prefix, reparam=True, embed_dim=512, mid_dim=512, float16=False, init_by_real_act=False):
        """
        Inputs:
        num_prefix: number of prefix tokens
        reparam: use reparameterization trick (not used in MeZO)
        embed_dim, mid_dim: hyperparameters for reparameterization trick (not used in MeZO)
        float15: whether the model parameters are float15
        init_by_real_act: init prefix tokens by real activations
        """

        self.model = model
        self.num_prefix = num_prefix 
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        # Reparameterization 
        self.reparam = reparam
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim

        input_embeds = None # For reparameterization
        if model.config.model_type == "opt":
            attention_name = "attn"
            first_layer_name = "layers.0"
            layer_name = "layers."
        elif model.config.model_type == "roberta":
            attention_name = "attention"
            first_layer_name = "layer.0"
            layer_name = "layer."
        elif model.config.model_type == "llama":
            attention_name = "self_attn"
            first_layer_name = "layers.0"
            layer_name = "layers."
        else:
            raise NotImplementedError

        if init_by_real_act:
            # Initialize prefix with real words' activations
            assert not reparam

            # Randomly sample input tokens
            input_tokens = torch.randint(low=0, high=model.config.vocab_size, size=(1, num_prefix), dtype=torch.long).cuda()
            if model.config.model_type in ["opt", "llama"]:
                with torch.no_grad():
                    # Get the real activations
                    real_key_values = model(input_ids=input_tokens, use_cache=True).past_key_values
            else:
                raise NotImplementedError   

        # Insert prefix
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                layer_id = int(key.split(layer_name)[1].split(".")[0])
                logger.info(f"Inject prefix to: {key}")
                _, _, attn = find_module(model, key)

                # Replace the old forward functions
                attn.original_forward = attn.forward
                attn.forward = attn_forward_hook.__get__(attn, type(attn))
                if not hasattr(attn, "num_heads"):
                    attn.num_heads = model.config.num_attention_heads
                first = first_layer_name in key
                self.add_prefix(attn, first=first, input_embeds=input_embeds)

                if first and self.reparam:
                    input_embeds = attn.prefix_input_embeds
                if init_by_real_act:
                    logger.info(f"Reinitialize with actual activation: {key} (layer {layer_id})")
                    keys = real_key_values[layer_id][0].squeeze(0).transpose(0, 1).reshape(num_prefix, -1)
                    values = real_key_values[layer_id][1].squeeze(0).transpose(0, 1).reshape(num_prefix, -1)
                    attn.prefix_keys.data = keys.to(attn.prefix_keys.data.device)
                    attn.prefix_values.data = values.to(attn.prefix_values.data.device)

        # Freeze non-prefix parameters
        for n, p in model.named_parameters():
            if "prefix" not in n:
                p.requires_grad = False

        # Replace the old prepare_inputs_for_generation function 
        model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))


    def add_prefix(self, module, first, input_embeds=None):
        device = module.k_proj.weight.data.device
        module.num_prefix = self.num_prefix
        module.reparam = self.reparam
        if self.reparam:
            if first:
                # For the first layer we inject the embeddings
                logger.info("For prefix+reparameterization, inject the embeddings in the first layer.")
                module.prefix_input_embeds = nn.Parameter(torch.randn(self.num_prefix, self.embed_dim, device=device, dtype=self.model.dtype), requires_grad=True)
            else:
                assert input_embeds is not None
                module.prefix_input_embeds = input_embeds
            module.prefix_mlp_keys = nn.Sequential(
                nn.Linear(self.embed_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.hidden_dim)
            ).to(device)
            module.prefix_mlp_values = nn.Sequential(
                nn.Linear(self.embed_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.hidden_dim)
            ).to(device)
            if self.float16:
                module.prefix_mlp_keys = module.prefix_mlp_keys.half()
                module.prefix_mlp_values = module.prefix_mlp_values.half()
        else:
            module.prefix_keys = nn.Parameter(torch.randn(self.num_prefix, self.hidden_dim, device=device, dtype=self.model.dtype), requires_grad=True)
            module.prefix_values = nn.Parameter(torch.randn(self.num_prefix, self.hidden_dim, device=device, dtype=self.model.dtype), requires_grad=True)
