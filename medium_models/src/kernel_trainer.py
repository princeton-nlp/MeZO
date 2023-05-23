########## The following part was originally copied from Transformers' trainer (3.4.0) and then changed heavily to compute eNTKs.  ##########

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
The trainer for computing eNTKs
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler

from functorch import vmap, jvp, jacrev, make_functional_with_buffers

import transformers
from transformers.data.data_collator import DataCollator
from transformers.file_utils import is_torch_tpu_available

from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer import SequentialDistributedSampler
from transformers.trainer_utils import PredictionOutput, EvalPrediction
from transformers.utils import logging
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
import gc
from transformers.trainer_utils import TrainOutput

from src.linearhead_trainer import varsize_tensor_all_gather, LinearHeadTrainer

import numpy as np
from tqdm import tqdm
from src.kernel_solvers import SOLVERS

logger = logging.get_logger(__name__)


class LogitModelWrapper(nn.Module):
    def __init__(self, model, binary_classification):
        super().__init__()
        self.model = model
        self.binary_classification = binary_classification

    def forward(self, input_ids, attention_mask, mask_pos):
        logits = self.model(input_ids, attention_mask, mask_pos=mask_pos)[0] # don't provide labels
        if self.binary_classification:
            assert logits.size(1) == 2, "--binary_classification should have 2 logits"
            logits = (logits[:,1] - logits[:,0]).unsqueeze(-1)
        return logits
            # label = (label * 2 - 1).float()  # convert from {0, 1} to {-1, 1}


def param_to_buffer(module, module_name, predicate):
    """Turns all parameters of a module into buffers."""
    modules = module.named_modules(prefix=str(module_name))
    next(modules) # Skip itself

    params = []
    for name, param in module.named_parameters(recurse=False, prefix=str(module_name)):
        if predicate(name):
            params.append((name.split(".")[-1], param))

    for name, param in params:
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param)
    for name, module in modules:
        param_to_buffer(module, name, predicate)


class KernelTrainerFunc(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        *posargs,
        **kwargs
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, *posargs, **kwargs)

        self.grad_dim = None
        self.train_train_kernel = None
        self.train_targets = None
        self.num_labels = None

        self.kernel_formula = args.kernel_formula

        def convert_to_buffer(name):
            if args.exclude_embeddings:
                if "embed" in name:
                    logger.info("Excluding {}".format(name))
                    return True

            if args.exclude_head:
                if "head" in name:
                    logger.info("Excluding {}".format(name))
                    return True

            if args.only_biases:
                if "bias" not in name:
                    logger.info("Excluding {}".format(name))
                    return True

            if args.exclude_first_layers != -1:
                if 'layers' in name:
                    layer_num = re.search('layers.(.+?).[a-z]', name).group(1)
                    if int(layer_num) < args.exclude_first_layers:
                        return True
                    else:
                        return False
                elif 'embed_tokens' in name or 'embed_positions' in name:
                    return True

            if model.model_args.apply_lora:
                if name.startswith('roberta') and "lora" not in name:
                    logger.info("Excluding {}".format(name))
                    return True
            return False

        param_to_buffer(self.model, "", convert_to_buffer)


    def get_unshuffled_dataloader(self, dataset: Optional[Dataset] = None, sharded: bool = False, batch_size: Optional[int] = -1):
        if sharded and is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif sharded and self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        bs = self.args.per_device_eval_batch_size if batch_size == -1 else batch_size
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=bs,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def profile_memory(self):
        import gc
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

    def compute_kernel_inner(self, curried_fn, curried_jacobian_fn, grads_outer, dataset_inner):

        # Use per_device_eval_batch_size for outer loop (which is always the training dataset
        dataloader_inner = self.get_unshuffled_dataloader(dataset_inner, sharded=False, batch_size=self.args.per_device_eval_batch_size)
        kernel_blocks = []
        targets_inner = []

        for inputs_inner in tqdm(dataloader_inner, desc="Computing kernel inner"):

            for k, v in inputs_inner.items():
                if isinstance(v, torch.Tensor):
                    inputs_inner[k] = v.to(self.args.device)

            def get_ntk_slice(tangents):
                _, jvps = curried_fn(inputs_inner.get("input_ids"), inputs_inner.get("attention_mask"), inputs_inner.get("mask_pos"), tangents)
                return jvps

            if self.args.kernel_formula == "signgd":
                grads_inner = curried_jacobian_fn(inputs_inner.get("input_ids"), inputs_inner.get("attention_mask"), inputs_inner.get("mask_pos"))
                block = sum(torch.einsum('olw,ikw->olik', j1.sign().flatten(2).to(torch.float64), j2.sign().flatten(2).to(torch.float64)).cpu() for j1, j2 in zip(grads_outer, grads_inner))
            else:
                block = vmap(vmap(get_ntk_slice))(grads_outer).to(torch.float64).cpu() # N_outer x C_outer x N_inner x C_inner

            kernel_blocks.append(block.detach())
            label = inputs_inner.get("labels")
            if self.args.binary_classification:
                label = (label * 2 - 1).float()  # convert from {0, 1} to {-1, 1}
            targets_inner.append(label)

            # del grads_inner
            del block
            del inputs_inner

            torch.cuda.empty_cache()
            gc.collect()

        return (
            torch.cat(kernel_blocks, dim=2) if kernel_blocks else torch.tensor([]),
            torch.cat(targets_inner, dim=0) if targets_inner else torch.tensor([])
        )

    def compute_kernel_outer(self, dataset_outer, dataset_inner):
        # Use train_batch_size for outer loop (which is always the training dataset)
        dataloader_outer = self.get_unshuffled_dataloader(dataset_outer, sharded=True, batch_size=self.args.per_device_train_batch_size)

        model_wrapper = LogitModelWrapper(self.model, self.args.binary_classification)
        model_wrapper.eval()
        for param in model_wrapper.parameters():
            param.requires_grad_(True)

        model_fn, params, buffers = make_functional_with_buffers(model_wrapper)

        jacobian_fn = jacrev(model_fn)

        def curried_jacobian_fn(input_ids, attention_mask, mask_pos):
            return jacobian_fn(params, buffers, input_ids, attention_mask, mask_pos)

        def curried_fn(input_ids, attention_mask, mask_pos, tangent):
            def curried_model_fn(params_):
                return model_fn(params_, buffers, input_ids, attention_mask, mask_pos)
            return jvp(curried_model_fn, (params,), (tangent,))

        kernel_rows = []

        inner_targets = None

        for inputs_outer in tqdm(dataloader_outer, desc="Computing kernel outer"):
            for k, v in inputs_outer.items():
                if isinstance(v, torch.Tensor):
                    inputs_outer[k] = v.to(self.args.device)

            grads_outer = curried_jacobian_fn(inputs_outer.get("input_ids"), inputs_outer.get("attention_mask"), inputs_outer.get("mask_pos"))
            if self.args.kernel_formula == 'asymmetric_signgd':
                grads_outer = tuple(g.sign() for g in grads_outer)

            # assert len(tuple(model_wrapper.model.named_parameters())) == len(grads_outer)

            # for (name,param), grad in zip(model_wrapper.named_parameters(), grads_outer):
            #     print(name, grad[(grad != 0).all(-1)])
            #     assert param.shape == grad.shape[2:], f"{name} {param.shape} {grad[2:].shape}"
            #     if (grad == 0).all():
            #         print(name, param.numel())

            if self.grad_dim is None:
                self.grad_dim = sum(np.prod(x.shape[2:]) for x in grads_outer)
            # assert self.grad_dim == num_params, "gradient dim not constant: {} and {}".format(self.grad_dim, num_params)

            kernel_blocks, inner_targets = (
                self.compute_kernel_inner(curried_fn, curried_jacobian_fn, grads_outer, dataset_inner))

            kernel_rows.append(kernel_blocks)

            del grads_outer
            del inputs_outer
            del kernel_blocks

            torch.cuda.empty_cache()
            gc.collect()


        kernel = torch.cat(kernel_rows, dim=0)

        return (
            kernel,
            inner_targets
        )

    def compute_kernel_sharded(self, dataset_outer, dataset_inner):
        assert self.kernel_formula in ["sgd", "asymmetric_signgd", "signgd"], "only sgd and asymmetric_signgd are supported by torchfunc for now"

        with torch.no_grad():
            kernel, inner_targets = self.compute_kernel_outer(dataset_outer, dataset_inner)

        if self.args.local_rank != -1:
            logger.info("Starting to gather kernel across GPUs")
            kernel = varsize_tensor_all_gather(kernel.to(self.args.device), torch.distributed.get_world_size())
            logger.info("Finished gathering kernel across GPUs")

        return kernel, inner_targets

    def compute_model_logits_cached(self, eval_dataset):
        if self.args.load_kernels is not None:
            output_dir = self.args.load_kernels
        else:
            output_dir = self.args.output_dir
        logit_file_name = f"{eval_dataset.mode}_logits_{eval_dataset.task_name}.pt"
        logit_path = os.path.join(output_dir, logit_file_name)

        if os.path.exists(logit_path) and not self.args.overwrite_kernels:
            logger.info(f"Starting to load logits from {logit_path}.")
            logits, targets = torch.load(logit_path)
            logger.info(f"Finished loading logits from {logit_path}.")
        else:
            logger.info(f"Starting to compute the {eval_dataset.mode} logits.")
            dataloader = self.get_unshuffled_dataloader(eval_dataset)

            model_wrapper = LogitModelWrapper(self.model, self.args.binary_classification)
            model_wrapper.eval()

            logits = []
            targets = []
            with torch.no_grad():
                for inputs in dataloader:
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.args.device)

                    label = inputs.get("labels")
                    if self.args.binary_classification:
                        label = (label * 2 - 1).float()  # convert from {0, 1} to {-1, 1}

                    preds = model_wrapper(inputs.get("input_ids"), inputs.get("attention_mask"), inputs.get("mask_pos"))
                    logits.append(preds.detach().cpu())
                    targets.append(label.cpu())

            logits = torch.cat(logits, dim=0)
            targets = torch.cat(targets, dim=0)

            logger.info(f"Finished computing the {eval_dataset.mode} logits.")

            if self.is_world_process_zero():
                torch.save((logits, targets), logit_path)
        return logits, targets

    def reshape_kernel_and_targets(self, kernel, targets):
        # reshape kernel to previous format
        if self.num_labels is None:
            self.num_labels = kernel.shape[1]
        assert self.num_labels == kernel.shape[1], "label dim not constant: {} and {}".format(self.num_labels, kernel.shape[1])
        assert self.num_labels == kernel.shape[3], "label dim not constant: {} and {}".format(self.num_labels, kernel.shape[3])

        if self.num_labels > 1: # multi logit
            targets = torch.nn.functional.one_hot(targets.squeeze(), self.num_labels)

        size1 = kernel.shape[0] * kernel.shape[1]
        size2 = kernel.shape[2] * kernel.shape[3]
        # kernel = kernel.transpose(0, 1).transpose(2, 3)
        return kernel.reshape(1, size1, size2), targets.reshape(-1)

    def compute_kernel_cached(self, eval_dataset):
        kernel_file_name = f"{eval_dataset.mode}_kernels_{eval_dataset.task_name}.pt"
        kernel_path = os.path.join(self.args.output_dir, kernel_file_name)

        if os.path.exists(kernel_path) and not self.args.overwrite_kernels:
            logger.info(f"Starting to load kernels from {kernel_path}.")
            (train_eval_kernel, eval_targets) = torch.load(kernel_path)
            logger.info(f"Finished loading kernels from {kernel_path}.")
        else:
            logger.info(f"Starting to compute the train-{eval_dataset.mode} kernel.")
            train_eval_kernel, eval_targets = self.compute_kernel_sharded(
                self.train_dataset, eval_dataset,
            )
            logger.info(f"Finshed computing the train-{eval_dataset.mode} kernel.")

            train_eval_kernel = train_eval_kernel.cpu()
            eval_targets = eval_targets.cpu()

            if self.args.kernel_formula == 'asymmetric_signgd':
                logger.info(f"Starting to compute the flipped train-{eval_dataset.mode} kernel.")
                if eval_dataset == self.train_dataset:
                    train_eval_kernel_flipped = train_eval_kernel
                else:
                    train_eval_kernel_flipped, _ = self.compute_kernel_sharded(
                        eval_dataset, self.train_dataset,
                    )
                logger.info(f"Finshed computing the flipped train-{eval_dataset.mode} kernel.")

                train_eval_kernel_flipped = train_eval_kernel_flipped.cpu()
                train_eval_kernel_flipped = train_eval_kernel_flipped.permute(2, 3, 0, 1)
                train_eval_kernel = torch.stack([train_eval_kernel, train_eval_kernel_flipped], dim=0)

            if self.is_world_process_zero():
                torch.save((train_eval_kernel, eval_targets), kernel_path)
        return train_eval_kernel, eval_targets


    def train(self, model_path=None, dev_objective=None):
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        if self.args.load_kernels is None:
            eval_dataset = self.train_dataset
            self.train_train_kernel, self.train_targets = self.compute_kernel_cached(eval_dataset)

        return TrainOutput(0, 0.0, {}), None

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if self.args.load_kernels is not None:
            logger.info(f"Starting to load kernels from {self.args.load_kernels}.")
            kernel_file_name = f"{eval_dataset.mode}_kernels_{eval_dataset.task_name}.pt"
            load_kernel_path = os.path.join(self.args.load_kernels, kernel_file_name)
            (train_eval_kernel, eval_targets) = torch.load(load_kernel_path)

            kernel_file_name = f"{self.train_dataset.mode}_kernels_{self.train_dataset.task_name}.pt"
            load_kernel_path = os.path.join(self.args.load_kernels, kernel_file_name)
            (self.train_train_kernel, self.train_targets) = torch.load(load_kernel_path)
            logger.info(f"Finished loading kernels from {self.args.load_kernels}.")
        else:
            assert self.train_train_kernel is not None, "train_train_kernel is None, did you forget to call train()?"
            train_eval_kernel, eval_targets = self.compute_kernel_cached(eval_dataset)

        if self.args.kernel_formula == 'asymmetric_signgd':
            train_eval_kernel_flipped = train_eval_kernel[1]
            train_eval_kernel_flipped, _ = self.reshape_kernel_and_targets(train_eval_kernel_flipped, eval_targets)

            train_eval_kernel = train_eval_kernel[0]
            train_train_kernel = self.train_train_kernel[0]
        else:
            train_eval_kernel_flipped = None
            train_train_kernel = self.train_train_kernel

        train_train_kernel, train_targets = self.reshape_kernel_and_targets(train_train_kernel, self.train_targets)
        train_eval_kernel, eval_targets = self.reshape_kernel_and_targets(train_eval_kernel, eval_targets)

        # get train and test logits
        if self.args.adjust_for_init:
            train_logits, _ = self.compute_model_logits_cached(self.train_dataset)
            eval_logits, _ = self.compute_model_logits_cached(eval_dataset)
            train_logits = train_logits.reshape(-1, 1)
            eval_logits = eval_logits.reshape(-1, 1)
        else:
            train_logits, eval_logits = None, None

        metrics = {}

        solver = SOLVERS[self.args.kernel_solver](self.args)
        solver.fit(train_train_kernel, train_targets, train_logits)
        eval_error, eval_preds = solver.predict(train_eval_kernel, eval_targets, eval_logits, eval_kernel_flipped=train_eval_kernel_flipped)
        eval_preds = eval_preds.reshape(-1, self.num_labels)
        if self.num_labels > 1:
            eval_targets = eval_targets.reshape(-1, self.num_labels).argmax(-1)

        if self.args.binary_classification: # Make sure to compute loss before this transformation!
            eval_preds = torch.cat([-eval_preds, eval_preds], dim=-1) # convert back to two logits
            eval_targets = ((eval_targets + 1) / 2).long() # convert back from {-1, 1} to {0, 1}

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=eval_preds.numpy(), label_ids=eval_targets.numpy()))

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        metrics["eval_loss"] = eval_error

        metrics.update(solver.metrics())
        metrics["grad_dim"] = self.grad_dim

        output = PredictionOutput(predictions=eval_preds.numpy(), label_ids=eval_targets.numpy(), metrics=metrics)
        self.log(output.metrics)

        return output
