# MeZO on Large Autoregressive Language Models

This part of the code is for MeZO experiments on large autoregressive language models. It includes training autoregressive LMs with linear probing, head tuning, full fine-tuning, parameter-efficient fine-tuning (PEFT), and MeZO. It also covers zero-shot and in-context learning (ICL) evaluation. It is tested on OPT-13B, 30B, and 66B but should be able to extend to other sizes and other autoregressive LMs.


## Installation

Please install the latest versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)), Transformers (`transformers`), and Accelerate (`accelerate`). This code is tested on `torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, and `accelerate==0.17.1` with Python 3.9.7, but should work with older/later versions of these packages too.

## Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning/MeZO):
```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below. 
* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning), or `zo` (MeZO).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: MeZO hyperparameter epsilon.
* `--prefix_tuning`: use prefix-tuning. 
* `--lora`: use LoRA.

We also support all [HuggingFace trainer arguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) for easily setting fine-tuning hyperparameters.

We provide example scripts below for reproducing our experiments. All our examples sample 1,000 training examples, 500 validation examples, and 1,000 testing examples. For ICL, we use 32 demonstrations. For detailed hyperparameters and grid search configs, please refer to Appendix D of [our paper](https://arxiv.org/pdf/2305.17333.pdf).
```bash
# Zero-shot
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0

# In-context learning
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh 

# Full-parameter fine-tuning, prefix-tuning, and LoRA
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash finetune.sh
MODEL=facebook/opt-1.3b TASK=SST2 MODE=prefix LR=1e-2 bash finetune.sh
MODEL=facebook/opt-1.3b TASK=SST2 MODE=lora LR=1e-4 bash finetune.sh

# Full-parameter fine-tuning using fully-sharded data parallel or FSDP (multi-GPU)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 NUM_GPU=4 bash finetune_fsdp.sh

# MeZO (full-parameter, prefix-tuning, and LoRA)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-13b TASK=SST2 MODE=prefix LR=1e-3 EPS=1e-1 bash mezo.sh
MODEL=facebook/opt-13b TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2 bash mezo.sh

# MeZO with non-differentiable objective (SQuAD (F1) + MeZO prefix as an example)
MODEL=facebook/opt-13b TASK=SQuAD MODE=prefix LR=1e-2 EPS=1e-1 bash mezo.sh --non_diff --evaluation_strategy no --save_strategy no --save_model
```

Note that `icl.sh` and `mezo.sh` automatically support multi-GPU usage. For fine-tuning, use `finetune_fsdp.sh` for multi-GPU training and specific `NUM_GPU`. Evaluation results (json format) and checkpoints (HuggingFace format) will be saved in `result` folder.

Our recommended hyperparameter search range for OPT-13b (should also work for other sizes/models) are as follows,

| MeZO methods  | LR           | EPS |
| ------------- | ------------ | --- |
| Full parameter  | 1e-6/1e-7 | 1e-3 |
| Prefix-tuning  | 1e-2/1e-3 | 1e-1 |
| LoRA  | 1e-4/5e-5  | 1e-2 |

## How to add MeZO to my own code?

Our implementation of MeZO is based on [HuggingFace Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py). We try to add MeZO to the official implementation of trainer with minimum editing. Please refer to `trainer.py` for details. We edit the `_inner_training_loop` function (to see where we edited, search `MeZO added`; ignore the linear probing part) to replace the original optimizer with MeZO, which contains the following operations: 

* `zo_perturb_parameters`: our in-place parameter perturbation function.
* `zo_forward` and `zo_forward_nondiff`: get loss value (differentiable/non-differentiable).
* `zo_step`: estimate the gradient by MeZO
* `zo_update`: update parameters by the estimated gradient.

To incorporate MeZO in your own HuggingFace code, simply overload HuggingFace's `Trainer` class and edit `_inner_training_loop` as we did in `trainer.py`, and add the above MeZO functions from `trainer.py` in your trainer. For a more intuitive explanation of the MeZO algorithm, please refer to Algorithm 1 in [our paper](https://arxiv.org/pdf/2305.17333.pdf). 
