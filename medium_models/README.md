# MeZO on Medium-sized Masked Language Models

This part of the code is for MeZO experiments on RoBERTa-large.

## Installation
Follow the installation and data preparation instructions in [this fine-tuning repository](https://github.com/princeton-nlp/LM-Kernel-FT) based on LM-BFF.

**NOTE**: Different versions of some packages (`pytorch`, `numpy`, `transformers`) may cause minor variations in results.

## Usage
`run.py` runs ZO training (and standard fine-tuning) of OPT-family and RoBERTa-family models. To use MeZO with SGD, add the flags `--zero_order_optim` and `--efficient_zero_order`. You can either specify the number of gradient steps desired or the number of forward passes allowed. Training will terminate when one of the two conditions is reached. Please refer to `*.sh.example` for script examples.

## Ablations
RoBERTa-large models can be fine-tuned on most single GPUs, so we did not yet implement all of the memory-efficient ZO variants discussed in Appendix B. We will release these features shortly. For now, if you want to run ablations other ZO ablations, you can add the flag `--zero_order_use_trainer_optim`, which will store the ZO gradients in the `param.grad` buffer and then use a PyTorch optimizer as usual. This causes the total memory consumption for ZO to be twice that of inference, which is still substantially less than that of backpropagation. The ablations can then be run with the additional following flags: 
- ZO-Adam: `--optimizer "adam"`
- ZO-Momentum: `--momentum <beta>`
- $n$-SPSA with $n>1$: `--zero_order_sample <n>` and you can add a linear or constant scheduler on it with `--zero_order_sample_scheduler {"linear", "constant"}`
- No prompt: `--few_shot_type finetune`

Appendix B discusses variants of ZO that modify the expectation and the variance. To run those one can use the following flags.
- Modify variance: `--zo_variant {"grad_norm", "param_norm"}`
- Recompute the control variate at the start of each epoch: `--recmopute_norms`
- Modify expectation: `--change_grad_estimate`
