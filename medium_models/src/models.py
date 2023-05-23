"""Custom models for few-shot learning specific operations."""

from socket import ntohl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
# from transformers.models.roberta.modeling_roberta import  RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from .modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
# from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTModel, OPTDecoder
from .modeling_opt import OPTPreTrainedModel, OPTModel, OPTDecoder, OPTForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Callable, Dict, Optional, Union, List, Tuple
import random


import logging
logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError

def model_for_prompting_forward(
    model,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    mask_pos=None,
    labels=None,
    sfc_input_ids=None,
    sfc_attention_mask=None,
    sfc_mask_pos=None
):
    if sfc_input_ids is not None:
        with torch.no_grad():
            logits = model_for_prompting_forward(model, input_ids=sfc_input_ids, attention_mask=sfc_attention_mask, mask_pos=sfc_mask_pos)[0]
        icl_sfc_bias = F.log_softmax(logits.detach().squeeze(0))

    if mask_pos is not None:
        mask_pos = mask_pos.squeeze()

    model_fn = model.get_model_fn()
    # Encode everything
    if token_type_ids is not None:
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    else:
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
        )


    # Get <mask> token representation
    sequence_output = outputs[0]
    if mask_pos is not None:
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
    else:
        sequence_mask_output = sequence_output[:,0] # <cls> representation
        # sequence_mask_output = sequence_output.mean(dim=1) # average representation

    if model.label_word_list is not None:
        # Logits over vocabulary tokens
        head_fn = model.get_lm_head_fn()
        prediction_mask_scores = head_fn(sequence_mask_output)

        # Exit early and only return mask logits.
        if model.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        # use MLM logit
        if model.model_args.use_task_word:
            vocab_logits = model.lm_head(sequence_mask_output)
            for _id in model.label_word_list:
                logits.append(vocab_logits[:, _id].unsqueeze(-1))
        # use learned linear head logit on top of task word representation (standard LM-BFF)
        else:
            for label_id in range(len(model.label_word_list)):
                logits.append(prediction_mask_scores[:, model.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if model.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity
    else:
        logits = model.classifier(sequence_mask_output)


    loss = None
    if labels is not None:
        if model.config.num_labels == 1:
            # Regression task
            if model.label_word_list is not None:
                labels = torch.stack([1 - (labels.view(-1) - model.lb) / (model.ub - model.lb), (labels.view(-1) - model.lb) / (model.ub - model.lb)], -1)
                loss = nn.KLDivLoss(log_target=True)(logits.view(-1, 2), labels)
            else:
                labels = (labels.float().view(-1) - model.lb) / (model.ub - model.lb)
                loss =  nn.MSELoss()(logits.view(-1), labels)
        else:
            if model.model_args.l2_loss:
                coords = torch.nn.functional.one_hot(labels.squeeze(), model.config.num_labels).float()
                loss =  nn.MSELoss()(logits.view(-1, logits.size(-1)), coords)
            else:
                loss =  nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))


    if hasattr(model, "lr_weight"):
        # Linear head
        logits = torch.matmul(F.softmax(logits, -1), model.lr_weight) 
    if hasattr(model, "lr_bias"):
        logits += model.lr_bias.unsqueeze(0)

    if model.model_args.sfc and hasattr(model, "sfc_bias"):
        logits = F.log_softmax(logits, -1) - model.sfc_bias
    if sfc_input_ids is not None:
        logits = F.log_softmax(logits, -1) - icl_sfc_bias

    output = (logits,)


    if model.model_args.use_task_word and model.num_labels == 1:
        # Regression output
        output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (model.ub - model.lb) + model.lb,)
    return ((loss,) + output) if loss is not None else output

def convert_opt_model(model: OPTModel, config, num_exclude_layers):
    model.model.decoder = EfficientOPTDecoder(config, num_exclude_layers)
    return model

class BertModelForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

        # For regression
        self.lb, self.ub = 0.0, 1.0

        # For auto label search.
        self.return_full_softmax = None

    def get_model_fn(self):
        return self.bert

    def get_lm_head_fn(self):
        return self.cls

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return model_for_prompting_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)

class RobertaModelForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        logger.warn("By default for RoBERTa models the input embeddings and the output embeddings are NOT tied!!!!")
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

        # For regression
        self.lb, self.ub = 0.0, 1.0

        # For auto label search.
        self.return_full_softmax = None
    
    def tie_emb(self):
        output_embeddings = self.lm_head.decoder
        self._tie_or_clone_weights(output_embeddings, self.roberta.get_input_embeddings())

    def get_model_fn(self):
        return self.roberta

    def get_lm_head_fn(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return model_for_prompting_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)

class OPTModelForPromptFinetuning(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        logger.warn("By default for OPT models the input embeddings and the output embeddings are tied!!!!")
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        # self.model = OPTModel(config)
        # self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

        # For regression
        self.lb, self.ub = 0.0, 1.0

        # For auto label search.
        self.return_full_softmax = None

    def get_model_fn(self):
        return self.model

    def get_lm_head_fn(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None, sfc_input_ids=None, sfc_attention_mask=None, sfc_mask_pos=None):
        return model_for_prompting_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels, sfc_input_ids=sfc_input_ids, sfc_attention_mask=sfc_attention_mask, sfc_mask_pos=sfc_mask_pos)

class GPT2ModelForPromptFinetuning(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        raise NotImplementedError("Need to check if the lm head is properly loaded and whether it is tied.")
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        # self.transformer = GPT2Model(config)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

        # For regression
        self.lb, self.ub = 0.0, 1.0

        # For auto label search.
        self.return_full_softmax = None

    def get_model_fn(self):
        return self.transformer

    def get_lm_head_fn(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return model_for_prompting_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)

class EfficientOPTDecoder(OPTDecoder):
    def __init__(self, config, num_exclude):
        super().__init__(config)
        self.num_exclude = num_exclude # number of initial layers to operate with torch.no_grad
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        #### run first few layers without grad
        with torch.no_grad():
            for idx, decoder_layer in enumerate(self.layers[:self.num_exclude]):

                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        None,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        #### rest of the layers with grad
        for idx, decoder_layer in enumerate(self.layers[self.num_exclude:]):
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        None,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


MODEL_TYPES = {
    "bert": BertModelForPromptFinetuning,
    "roberta": RobertaModelForPromptFinetuning,
    "opt": OPTModelForPromptFinetuning,
    "gpt2": GPT2ModelForPromptFinetuning,
}
