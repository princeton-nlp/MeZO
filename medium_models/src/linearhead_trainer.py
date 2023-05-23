########## The following part was originally copied from Transformers' trainer (3.4.0) and then changed heavily for linear head probing.  ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A trainer for finding linear probing solutions
"""

import collections
from src.models import MODEL_TYPES
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

import transformers
from torch.utils.data.dataset import Dataset
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


def tensor_all_gather(tensor: torch.Tensor, distributed_world_size: int):
    tensor_list = [torch.zeros_like(tensor) for _ in range(distributed_world_size)]
    torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor)
    return torch.cat(tensor_list, dim=0)


def varsize_tensor_all_gather(tensor: torch.Tensor, distributed_world_size: int):
    tensor = tensor.contiguous()

    dim_tensor = torch.tensor([tensor.size(0)], dtype=torch.int64, device=tensor.device)
    dim_tensor = tensor_all_gather(dim_tensor, distributed_world_size).cpu()
    max_size = dim_tensor.max()

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=tensor.device)
    padded[:tensor.shape[0]] = tensor

    ag = tensor_all_gather(padded, distributed_world_size)
    slices = []
    for i, sz in enumerate(dim_tensor):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    return torch.cat(slices, dim=0)

def get_token_prediction_layer(model):
    if isinstance(model, tuple(MODEL_TYPES.values())):
        if model.label_word_list is not None:
            lm_head = model.get_lm_head_fn()
            if model.model_type == "roberta":
                return lm_head.decoder
            elif model.model_type == "bert":
                return lm_head.predictions.decoder
            elif model.model_type == "gpt2" or model.model_type == 'opt':
                return lm_head # TODO: has no bias so linear regression with bias will fail right now
        else:
            return model.classifier
    elif isinstance(model, transformers.RobertaForSequenceClassification):
        return model.classifier.out_proj
    elif isinstance(model, transformers.BertForSequenceClassification):
        return model.classifier
    else:
        raise NotImplementedError(model.__class__)

def extract_features(model, *args, **kwargs):
    """some magic for getting features pre last layer"""
    features = {}
    def hook(model_, input_, output_):
        features["features"] = input_[0].detach()

    get_token_prediction_layer(model).register_forward_hook(hook)
    model.forward(*args, **kwargs)
    return features["features"]

def extract_features_prob(model, *args, **kwargs):
    """some magic for getting all logits"""
    output = model.forward(*args, **kwargs)
    return F.softmax(output[1].detach(), -1)



class LinearHeadTrainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        self.best_dir = None
        self.objective = -float("inf")

        model = self.model
        model.eval()

        train_dataloader = self.get_train_dataloader()
        targets = []
        features = []

        logger.info("Starting to get features for training dataset")
        with torch.no_grad():
            for step, inputs in enumerate(train_dataloader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                if self.args.prob_as_feature:
                    features.append(extract_features_prob(model, **inputs))
                else:
                    features.append(extract_features(model, **inputs))
                targets.append(inputs["labels"])
        logger.info("Finished getting features for training dataset")

        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)

        if self.args.local_rank != -1:
            logger.info("Starting to gather features across workers")
            features = varsize_tensor_all_gather(features, torch.distributed.get_world_size())
            targets = varsize_tensor_all_gather(targets, torch.distributed.get_world_size())
            logger.info("Finished gathering features across workers")

        features = features.cpu()
        targets = targets.cpu()

        if model.num_labels == 1:  # Regression
            targets_coords = targets.squeeze().unsqueeze(-1).float()
            reg = LinearRegression().fit(features.numpy(), targets_coords.numpy())
        else:
            use_bias = (model.model_type != 'opt' and model.model_type != 'gpt2') or self.args.prob_as_feature
            tol = 0.01 if self.args.lp_early_stopping else 1e-4            # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol).fit(features.numpy(), targets.numpy())
            # targets_coords = torch.nn.functional.one_hot(targets.squeeze(), model.num_labels).float()

        logger.info("Fitting linear regression")

        logger.info("Assigning weights to model")
        # print(head.out_proj.weight.shape, head.out_proj.bias.shape)
        # print(reg.coef_.shape, reg.intercept_.shape)
        decoder = get_token_prediction_layer(model)
        coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
        if use_bias:
            bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)

        if model.num_labels == 2 and coef_torch.size(0) == 1:
            coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
            if use_bias:
                bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

        if decoder.weight.shape[0] == model.num_labels:
            decoder.weight.data = coef_torch
            if use_bias:
                decoder.bias.data = bias_torch
        else:
            if self.args.prob_as_feature:
                model.lr_weight = coef_torch
                if use_bias:
                    model.lr_bias = bias_torch
            else:                
                decoder.weight.data[model.label_word_list,:] = coef_torch
                if use_bias:
                    decoder.bias.data[model.label_word_list] = bias_torch

        if model.num_labels == 1:  # Regression
            logits = torch.tensor(reg.predict(features.numpy()))
            train_loss = torch.nn.functional.mse_loss(logits, targets_coords, reduction="none")
        else:
            logits = torch.tensor(reg.predict_log_proba(features.numpy()))
            train_loss = torch.nn.functional.cross_entropy(logits, targets.squeeze(), reduction="none")

        return TrainOutput(0, train_loss, {}), self.objective



    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
