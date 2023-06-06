import json
import os
import contextlib
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
import logging
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import transformers
from typing import Optional, Union, List, Dict, Any
import signal
from subprocess import call
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase



logger = logging.getLogger(__name__)


def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits

    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None: 
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1) # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0) # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, max_new_tokens=None):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space 
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()
    
    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc; verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            # With demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # Tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]  
   
    return encodings, option_lens
 


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        
        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0, "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature
            
        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != "gold"} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]
        
        return batch
        

class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)