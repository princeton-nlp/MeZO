# MeZO on Medium-sized Masked Language Models

This part of the code is for MeZO experiments on RoBERTa-large. It is based on [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT) and [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

## Installation

Please install the latest versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)) and Transformers (`transformers`). This code is tested on `torch==2.1.0.dev20230514+cu118` and `transformers==4.28.1` with Python 3.9.7, but should work with older/later versions of these packages too.

## Prepare the data

We pack the datasets [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash download_dataset.sh
```

Then use the following command (in the `medium_models` folder) to generate the data we need:

```bash
for K in 16 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and `K=512` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot-1k-test`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples.

## Usage

Use `run.py` for all functions and refer to `run.py` for the usage of all arguments.
```bash
python run.py {ARGUMENTS}
```

To reproduce our results in the paper, we also provide two example files `finetune.sh` (for all fine-tuning experiments) and `mezo.sh` (for all MeZO experiments). You can run them directly with the following commands (we use the following six datasets in our experiments -- `SST-2`, `sst-5`, `SNLI`, `MNLI`, `RTE`, and `trec`):
```bash
# Adam fine-tuning
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh

# Adam fine-tuning + prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-2 MODEL=roberta-large EXTRA_TAG=prefix bash finetune.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# Adam fine-tuning + LoRA
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16

# MeZO
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-6 EPS=1e-3 MODEL=roberta-large bash mezo.sh

# MeZO + prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-2 EPS=1e-1 MODEL=roberta-large EXTRA_TAG=prefix bash mezo.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# MeZO + LoRA
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mezo.sh --apply_lora --lora_r 8 --lora_alpha 16
```
You can designate different hyperparameters by passing different environment variables as shown above. You can also directly add arguments at the end of the command to override the default ones. For all the hyperparameters you can control via environment variables, please refer to `finetune.sh` and `mezo.sh`. For the hyperparameters we used in our experiments, please refer to Appendix D of our paper.


## Gather results

All the results will be stored in `./log`. To analyze the results (for example, examine the grid search), use the following command
```bash
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-ft', 'task_name': 'sst-2'}"
```

Then the program will find all the trials that satisfy the condition in `./log`, and print the mean/std of the final results. Note that the task names are all lower-cased here.


## Ablations

RoBERTa-large models can be fine-tuned on most single GPUs, so we did not yet implement all of the memory-efficient ZO variants discussed in Appendix B. For now, if you want to run ablations other ZO ablations, you can add the flag `--zero_order_use_trainer_optim`, which will store the ZO gradients in the `param.grad` buffer and then use a PyTorch optimizer as usual. This causes the total memory consumption for ZO to be twice that of inference, which is still substantially less than that of backpropagation. The ablations can then be run with the additional following flags: 
- ZO-Adam: `--optimizer "adam"`
- ZO-Momentum: `--momentum <beta>`
- $n$-SPSA with $n>1$: `--zero_order_sample <n>` and you can add a linear or constant scheduler on it with `--zero_order_sample_scheduler {"linear", "constant"}`
- No prompt: `--few_shot_type finetune`

Appendix B discusses variants of ZO that modify the expectation and the variance. To run those one can use the following flags.
- Modify variance: `--zo_variant {"grad_norm", "param_norm"}`
- Recompute the control variate at the start of each epoch: `--recmopute_norms`
- Modify expectation: `--change_grad_estimate`
