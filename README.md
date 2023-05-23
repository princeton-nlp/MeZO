# A Kernel-Based View of Language Model Fine-Tuning

## Documentation for Alex and Eshaan
Follow the installation and data preparation instructions below. Install W&B on top of it and make sure you're logged in. `run_fewshot.sh` has the script for running the code - you will need to specify a `TAG` that you can search by later to find the results. So you could for example do 
```
TAG=debug WANDB_MODE=online WANDB_PROJECT=your-project TASK=SNLI MODEL=roberta-large K=256 run_fewshot.sh --zero_order_optim --zero_order_eps 1e-3 --learning_rate 1e-6 --zero_order_use_trainer_optim
```
We have many tasks implemented. I would say SNLI with RoBERTa-large, because it's not too hard but not too easy either...it's good practice to eventually do things for all 5 seeds, but when you're debugging you can just use one seed. `K=256` is what I've been using for the smaller models.

Reference numbers with K=256:
- Zero-shot: 50.2
- Linear probing (freeze model, learn head on representations): 77.9 (4.3)
- Adam FT (grid searched): 87.0 (5.3)
- ZO-SGD: 74.0 (2.9)
- ZO-SGD with best HPs + large budget: 79.1 (3.2)
- ZO-Adam: 71.7 (2.8)

Here are the relevant flags:
- `--zero_order_optim`: uses ZO instead of FT
- `--zero_order_eps`: the epsilon to use in the ZO algorithm
- `--zero_order_use_trainer_optim`: uses the PyTorch trainer instead of applying gradients manually. Less memory efficient but compatible with all the other opt flags like `--weight_decay`.
- `--zero_order_sample`: number of zs to use at each gradient step
- `--zero_order_sample_scheduler`: {"linear, "power", "constant"}. When this is set to "linear", `--zero_order_sample` specifies the max number of samples.
- `--scale_lr_with_samples`: scales the LR with the number of samples of z. Instead of using the average of the projected gradients, just directly uses the sum
- `--preconditioned_zero_order`: runs the preconditioned ZO algorithm. Uses backprop in the beginning to estimate the scales of the gradients. 

## Grid search

Copy the example file you want to use and edit the grid, then to run grid search, execute the following bash (modify NUM_GRID_SEARCH to your number of grid search experiments)
```bash
for TASK in SST-2 SNLI trec MRPC
do
sbatch --output=slurm/%A_%a-%x.out -N 1 -n 1 --cpus-per-task=5 --mem=20G --gres=gpu:rtx_2080:1 --mail-type=FAIL,TIME_LIMIT  -A allcs --time 1:0:0 --array 0-{NUM_GRID_SEARCH-1} --job-name sometag-$TASK  <<EOF
#!/bin/bash
TASK=$TASK bash opt_fewshot_zero_order_adam_array.sh
EOF
done
```

## In-context learning
```bash
SEED=$SEED TYPE=prompt-demo GRID_TAG=$TAG STEPS=0 TASK=$TASK TAG="icl-32" MODEL=$MODEL K=16 \
bash run_zeroshot_opt.sh --per_device_eval_batch_size 2 --gpt3_in_context_head --gpt3_in_context_num 32 --truncate_head --auto_demo --num_sample 1 --efficient_zero_order --efficient_zero_order_fp16
```

-------

This is the implementation for the paper [A Kernel-Based View of Language Model Fine-tuning](https://arxiv.org/abs/2210.05643)
and can be used to compute kernel approximations for the fine-tuning of pre-trained language models.

We extend the [LM-BFF](https://github.com/princeton-nlp/LM-BFF) repository and
add a new "kernel trainer" powered by [functorch](https://github.com/pytorch/functorch) to compute empirical-NTK kernel matrices using the SGD, SignGD or Asymmetric-SignGD kernel formulas.
We also provide our pre-computed kernels for download to facilitate further analysis.

## Installation
Please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

We updated the LM-BFF code to work with a newer version of HuggingFace transformers and additionally require functorch.
If you would like to run LoRA fine-tuning, install the LoRA version of the transformers library ([see here](https://github.com/microsoft/LoRA/tree/main/examples/NLU)) and add the flags `--apply_lora --lora_alpha .... --lora_r ...` .

**NOTE**: Different versions of some packages (`pytorch`, `numpy`, `transformers`) may cause minor variations in kernels and results.

## Prepare the data
Please run the following commands to download and prepare the data:

```bash
( cd data; bash download_dataset.sh )

for K in 16 64 256 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

This follows LM-BFF, but `download_dataset.sh` additionally rebalances the `cr` dataset and uses the GLUE version of the SST-2 dataset. Additionally `k-shot-1k-test` limits test datasets to 1k examples for faster evaluation.

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run the code
To easily run our experiments, you can use `run_fewshot.sh`:

```bash
TAG=kernel-prompting TRAINER=kernel TASK=SST-2 SEED=42 MODEL=roberta-base bash run_fewshot.sh
```

The templates and label word mappings are already defined, so you only need to set hyper-parameters and `TAG` (you can use whatever tag you want and it just makes finding results easier). See `run_fewshot.sh` for more options. Besides, you can easily add extra arguments:

```bash
NUM_GPU=4 TAG=kernel-prompting TRAINER=kernel TASK=SST-2 SEED=42 MODEL=roberta-base bash run_fewshot.sh \
    --kernel_formula signgd --kernel_solver logistic  --per_device_train_batch_size 2 --per_device_eval_batch_size 4
```
This splits the kernel computation across 4 GPUs and uses the SignGD kernel formula and a logistic kernel solver (the default is least-squares regression) and uses batch sizes 2 and 4 along the two axes of the kernel matrices respectively.

For more advanced use cases, such as [how to aggregate results over multiple runs](https://github.com/princeton-nlp/LM-BFF#experiments-with-multiple-runs), [zero-shot experiments](https://github.com/princeton-nlp/LM-BFF#zero-shot-experiments) or [writing your own prompt formats](https://github.com/princeton-nlp/LM-BFF#how-to-design-your-own-templates), we refer to the README in the LM-BFF repo.
Note that we deleted some tools to do automatic prompt and label search that are unrelated to our paper.

 ## Download our pre-computed kernels
Here are the links for downloading our pre-computed kernels:
* [SGD kernels](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/sgd.zip)
* [SignGD kernels](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/signgd.zip)
* [Asymmetric-SignGD kernels](https://nlp.cs.princeton.edu/projects/LM-Kernel-FT/roberta-base/asymmetric_signgd.zip)

The provided kernels were computed with RoBERTa-base for 12 datasets (SST-2, MR, CR, MPQA, Subj, TREC, MNLI, SNLI, QNLI, RTE, MRPC, QQP) over 5 seeds on both 16-shot and 64-shot datasets, where k-shot is the number of training/validation examples per label.
The SGD kernels also include 6 datasets (SST-2, MR, CR, QNLI, RTE, QQP) for 512-shot datasets.

For each task and data split, we include separate files for training, development, test kernel matrices. Each file can be read using `torch.load` and contains a tuple of (kernel matrix, labels),
and the kernel matrix has the shape of [training examples, training logits, X examples, X logits], where X dataset is given by the file name (train, dev or test).

## Bugs and questions?
If you have any questions related to the code or the paper, feel free to email Alexander and Sadhika (`{awettig,smalladi}@cs.princeton.edu`). If you encounter a problem or bug when using the code, you can also open an issue.

## Citation

Please cite our work if you make use of our code or our pre-computed kernels in your work:

```bibtex
@article{malladi2022kernel,
      title={A Kernel-Based View of Language Model Fine-Tuning},
      author={Malladi, Sadhika and Wettig, Alexander and Yu, Dingli and Chen, Danqi and Arora, Sanjeev},
      journal={arXiv preprint arXiv:2210.05643},
      year={2022}
}
```
