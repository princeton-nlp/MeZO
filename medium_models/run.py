"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List
import torch
import torch.nn.functional as F

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, PreTrainedTokenizerBase
from src.modeling_roberta import RobertaConfig
from src.modeling_opt import OPTConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.linearhead_trainer import LinearHeadTrainer
from src.dataset import FewShotDataset, OurInputFeatures
from src.models import MODEL_TYPES, resize_token_type_embeddings, convert_opt_model
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    l2_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use L2 loss (only makes a difference in standard FT)."}
    )
    use_task_word: bool = field(
        default=False,
        metadata={'help': 'uses the task words MLM logit for kernel computation'}
    )

    # LoRA arguments: only for BERT-type model
    apply_lora: bool = field(
        default=False,
        metadata={'help': 'use LoRA for finetuning'}
    )
    lora_alpha: int = field(
        default=None,
        metadata={'help': 'initialization scale for one of the low rank matrices in lora'}
    )
    lora_r: int = field(
        default=None,
        metadata={'help': 'inner rank for lora matrices'}
    )

    # Calibration
    sfc: bool = field(
        default=False,
        metadata={"help": "Whether to use surface form calibration."}
    )

    icl_sfc: bool = field(
        default=False,
        metadata={"help": "Use in-context learning demos in sfc."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    sfc_prompt: str = field(
        default=None,
        metadata={"help": "SFC prompt"}
    )

    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    gpt3_demo_separator: str = field(
        default="\n\n\n",
        metadata={"help": "Separator between demonstrations"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List[str] = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."},

    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training or at the."}
    )
    log_file: str = field(
        default='log'
    )

    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-parameter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    optimizer: str = field(
        default='adam',
        metadata={'help': 'choose sgd or adam. default is adam'}
    )
    optimizer_variant: str = field(
        default='',
        metadata={'help': 'define variants on optimizer: signgd'}
    )

    trainer: str = field(
        default="standard",
        metadata={"help": "Pick from {standard, kernel, linearhead}"}
    )
    from_linearhead: bool = field(
        default=False,
        metadata={"help": "Whether to initialize head with the linearhead solution. Works for both normal and kernel trainer."}
    )
    lp_early_stopping: bool = field(
        default=False,
        metadata={"help": "When on, increases the tolerance and lowers max_iter in scikit LogisticRegression solver to encourage early stopping."}
    )
    random_model_init: bool = field(
        default=False,
        metadata={'help': 'reinit the model randomly'}
    )
    sweep: bool = field(
        default=False,
        metadata={'help': 'configures the output directories to be informative when running W&B sweep'}
    )
    kernel_formula: str = field(
        default='sgd',
        metadata={"help": "choose kernel formula from {sgd, signgd, asymmetric_signgd}"}
    )
    kernel_solver: str = field(
        default="logistic",
        metadata={"help": "choose kernel solver from {lstsq, logistic, svr, svc, asym (only for asymmetric_signgd)}"}
    )
    load_kernels: str = field(
        default=None,
        metadata={'help': 'when specified, loads the kernels from the folder given here'}
    )
    overwrite_kernels: bool = field(
        default=False,
        metadata={'help': 'when specified, overwrites the kernels in the output_dir and computes them from scratch'}
    )

    exclude_embeddings: bool = field(
        default=False,
        metadata={"help": "Don't use embeddings for kernel computation "}
    )
    exclude_head: bool = field(
        default=False,
        metadata={"help": "Don't use head for kernel computation "}
    )
    only_biases: bool = field(
        default=False,
        metadata={"help": "Only use bias parameters for kernel computation for BitFit-style kernel"}
    )
    exclude_first_layers: int = field(
        default=-1,
        metadata={'help': 'excludes the first N layers from kernel computation'}
    )
    sync_embedding_layers: bool = field(
        default=False,
        metadata={'help': 'sync the input embedding to match output embedding (use with --exclude_first_layers)'}
    )

    kernel_regularization: float = field(
        default=0.0,
        metadata={"help": "Regularization constant for kernel"}
    )
    kernel_gamma: float = field(
        default=1.0,
        metadata={"help": "Gamma for asymmetric kernel solver"}
    )
    binary_classification: bool = field(
        default=False,
        metadata={"help": "If num_classes=2, convert two softmax logits to single sigmoid logit"}
    )
    adjust_for_init: bool = field(
        default=False,
        metadata={'help': 'when on, trains kernel on y-f0 and adds f0 at test time'}
    )
    f0_scaling: float = field(
        default=1.0,
        metadata={'help': 'adjust label scaling, might help with --adjust_for_init perf'}
    )
    zero_order_optim: bool = field(
        default=False,
        metadata={'help': 'when on, trains the model by zero-order optimization'}
    )
    zero_order_eps: float = field(
        default=1e-3,
        metadata={'help': 'eps for zero order optim'}
    )
    prob_as_feature: bool = field(
        default=False,
        metadata={'help': 'in linear head, use log prob as feature'}
    )
    zero_order_use_trainer_optim: bool = field(
        default=False,
        metadata={"help": "Use trainer optimizer for zero order optimization"}
    )
    efficient_zero_order: bool = field(
        default=False,
        metadata={"help": "Efficient zero-order: resample noise vectors instead of saving them. enable different model loading using --hf_inference_model"}
    )
    hf_inference_model: bool = field(
        default=False,
        metadata={"help": "loads the HF model in inference mode across many GPUs. incompatible with --zero_order_use_trainer_optim."}
    )
    efficient_zero_order_fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 for efficient zero order"}
    )
    zero_order_sample_scheduler: str = field(
        default=None,
        metadata={"help": "Have a sample scheduler. None, 'linear', 'power', or 'constant."}
    )
    scale_lr_with_samples: bool = field(
        default=False,
        metadata={"help": "Scales the LR proportionally to the number of z samples. --learning_rate will be the LR for one z sample."}
    )
    zero_order_sample: int = field(
        default=1,
        metadata={"help": "Sample times for zero-order estimate. If scheduler is 'linear', this number is the max sample number."}
    )
    zero_order_clip_grad: bool = field(
        default=False,
        metadata={"help": "Clip the norm of the gradient for zero order (only when using trainer optimizer)"}
    )
     
    # MeZO variants
    zo_by_layer: bool = field(
        default=False,
        metadata={"help": "For ZO: estimate the gradients on each layer individually, scales number of forward passes per grad step by a factor of L"}
    )
    zo_variant: str = field(
        default=None,
        metadata={"help": "Choose the MeZO variant: grad_norm or param_norm (see documentation)"}
    )
    use_zo_grad_est: bool = field(
        default=False,
        metadata={"help": "Use zero-order estimate of the gradient for zo variants"}
    )
    recompute_norms: bool = field(
        default=False,
        metadata={'help': 'Recompute the grad or parameter norm (whichever was specified as --zo_variant) at the start of each epoch.'}
    )
    scale_norm_by_num_params: bool = field(
        default=False,
        metadata={'help': 'Scale grad or param norm by 1 / sqrt(num params)'}
    )
    norm_running_update: bool = field(
        default=False,
        metadata={"help": "When performing --zo_by_layer and using --zo_variant 'grad_norm', update the layer grad norms as they are recomputed at each step"}
    )
    change_grad_estimate: bool = field(
        default=False,
        metadata={"help": "Changes the expectation of the ZO gradient estimate according to zo_variant, instead of just modifying the variance"}
    )

    # prefix tuning hyperparameters
    prefix_tuning: bool = field(
        default=False,
        metadata={"help": "Prefix tuning"}
    )
    num_prefix: int = field(
        default=10,
        metadata={"help": "How many prefix tokens to use"}
    )
    no_reparam: bool = field(
        default=False,
        metadata={"help": "No reparameterization trick"}
    )
    prefix_init_by_real_act: bool = field(
        default=False,
        metadata={"help": "For no_reparam case, randomly sample words and take their actual key/value pairs as initialization"}
    )
    layer_wise_optim: bool = field(
        default=False,
        metadata={'help': 'Optimize layer-by-layer (only for prefix + ZO)'}
    )

    max_zo_forward_steps: int = field(
        default=0,
        metadata={'help': 'Stop at this number of ZO forward steps. The trainer will take whichever is reached first, max_steps or max_zo_forward_steps.'}
    )
    
    untie_emb: bool = field(
        default=False,
        metadata={"help": "Untie embeddings from lm head. Only work for OPT!!"}
    )
    tie_emb: bool = field(
        default=False,
        metadata={"help": "Tie embeddings from lm head. Only work for RoBERTa!!"}
    )
    
    optimize_acc: bool = field(
        default=False,
        metadata={"help": "Maximize accuracy instead of minimizing loss"}
    )


    ## hessian trainer args
    num_hvp_vecs: int = field(
        default=128,
        metadata={"help": "Number of vectors to use to estimate HVPs"}
    )
    mc_tol: float = field(
        default=0.1,
        metadata={"help": "Tolerance (on std dev) after which MC estimate is deemed converged"}
    )

    head_tuning: bool = field(
        default=False,
        metadata={"help": "Tune the head only"}
    )


@dataclass
class MyDataCollatorWithPadding:
    """
    Implements padding for LM-BFF inputs.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        mask_pos = []
        standard_features = []
        if features[0].sfc_input_ids is not None:
            sfc_batch = self.__call__([OurInputFeatures(input_ids=x.sfc_input_ids, attention_mask=x.sfc_attention_mask, mask_pos=x.sfc_mask_pos) for x in features])

        for item in features:
            standard_item = {}
            for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)
            mask_pos.append(item.mask_pos)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(mask_pos):
            batch["mask_pos"] = torch.tensor(mask_pos)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        if features[0].sfc_input_ids is not None:
            batch["sfc_input_ids"] = sfc_batch["input_ids"]
            batch["sfc_attention_mask"] = sfc_batch["attention_mask"]
            batch["sfc_mask_pos"] = sfc_batch["mask_pos"]
        return batch


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.sweep:
        now = datetime.now()
        dt_str = now.strftime('%m_%d_%H_%M_%S')
        training_args.output_dir = os.path.join(training_args.output_dir, dt_str)

    #if model_args.apply_lora:
    #    assert 'roberta' in model_args.model_name_or_path, 'LoRA only implemented for RoBERTa models'

    if training_args.kernel_formula == 'asymmetric_signgd':
        assert training_args.binary_classification, 'asymmetric solver not implemented for multi-class setting, use --binary_classification'

    if training_args.optimizer_variant != '':
        assert training_args.optimizer == 'sgd', 'variants on optimizer are only implemented for SGD'

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True


    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            new_sfc_template = data_args.sfc_prompt + ''
            old_template = old_template.replace('*cls*', '')
            old_template = old_template.replace('*bos*', '')
            if data_args.gpt3_in_context_head:
                new_template = new_template.replace('*cls*', '')
                new_template = new_template.replace('*bos*', '')

            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                if "opt" in model_args.model_name_or_path or "gpt" in model_args.model_name_or_path:
                    sub_template = sub_template + "*labelx_{}*".format(instance_id)
                else:
                    sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + data_args.gpt3_demo_separator + sub_template # Put context at the end
                    new_sfc_template = new_sfc_template + data_args.gpt3_demo_separator + sub_template
                else:
                    new_template = sub_template + data_args.gpt3_demo_separator + new_template # Put context at the beginning
                    new_sfc_template = sub_template + data_args.gpt3_demo_separator + new_sfc_template
            if data_args.gpt3_in_context_head:
                new_template = "*bos*" + new_template
                new_sfc_template = "*bos*" + new_sfc_template
            logger.info("| {} => {}".format(data_args.template, new_template))
            logger.info("New SFC template (in-context learning): {}".format(new_sfc_template))
            data_args.template = new_template
            if model_args.icl_sfc:
                data_args.icl_sfc_prompt = new_sfc_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config_kwargs = {'apply_lora': model_args.apply_lora,
                    'lora_alpha': model_args.lora_alpha,
                    'lora_r': model_args.lora_r}
    if model_args.apply_lora:
        if 'roberta' in model_args.model_name_or_path:
            config = RobertaConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                **config_kwargs)
        else:
            config = OPTConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                **config_kwargs
            )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir
        )

    if training_args.untie_emb:
        logger.warn("Untie embeddings and lm head")
        logger.warn("NOTE that this only works for OPT. By default RoBERTa model embeddings are already untied.")
        config.tie_word_embeddings = False

    if 'prompt' in model_args.few_shot_type:
        model_fn = MODEL_TYPES[config.model_type]
    elif model_args.few_shot_type == 'finetune':
        if training_args.from_linearhead:
            model_fn = MODEL_TYPES[config.model_type]
        else:
            model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )
    if "opt" in model_args.model_name_or_path:
        # Set SEP token
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = 0
    if "gpt2" in model_args.model_name_or_path:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id


    if training_args.hf_inference_model:
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-5}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}

        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            device_map='auto',
            torch_dtype=torch.float16 if training_args.efficient_zero_order_fp16 else torch.float32,
            max_memory=max_memory,
        )
    else:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    if training_args.tie_emb:
        logger.warn("Tie embeddings. Only work for RoBERTa (in our code by default they are not tied)")
        model.tie_emb()
    
    if training_args.head_tuning:
        if model.config.model_type == "roberta":
            head_name = "lm_head"

        for n, p in model.named_parameters():
            if head_name not in n:
                p.requires_grad = False 
            else:
                logger.info(f"Only tuning {n}")        

    tokenizer.model_type = model.config.model_type

    if training_args.exclude_first_layers != -1:
        model = convert_opt_model(model, config, training_args.exclude_first_layers)

    if training_args.prefix_tuning:
        from src.prefix import PrefixTuning
        PrefixTuning(model, num_prefix=training_args.num_prefix, reparam=not training_args.no_reparam, float16=training_args.efficient_zero_order_fp16, init_by_real_act=training_args.prefix_init_by_real_act)

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_train
        else None
    )
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_predict
        else None
    )

    set_seed(training_args.seed)

    if training_args.random_model_init:
        model.init_weights() # reinit weights to random

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    if eval_dataset.label_word_list is not None:
        model.label_word_list = torch.tensor(eval_dataset.label_word_list).long().to(training_args.device)
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    if model_args.apply_lora:
        for name, param in model.named_parameters():
            if (name.startswith('roberta') and "lora" not in name) or (name.startswith('opt') and "lora" not in name):
                param.requires_grad_(False)

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]

            num_sample = test_dataset.num_sample if eval_dataset is None else eval_dataset.num_sample
            logits = predictions.reshape([num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer_classes = {
        "standard": Trainer,
        "linearhead": LinearHeadTrainer,
    }
    trainer_class = trainer_classes[training_args.trainer]
    trainer_kwargs = {}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=MyDataCollatorWithPadding(tokenizer),
        **trainer_kwargs
    )

    # Calibration
    if model_args.sfc:
        inputs = tokenizer([data_args.sfc_prompt.replace("_", " ")], return_tensors="pt")
        logger.info(f"Calibrating SFC with prompt: {data_args.sfc_prompt}")
        logger.info("Inputs: {}".format(inputs.input_ids))
        inputs = inputs.to(model.device)
        with torch.no_grad():
            model.eval()
            logits = model(**inputs)[0]
        model.sfc_bias = F.log_softmax(logits.squeeze(0).detach())
        logger.info("SFC bias: {}".format(model.sfc_bias))


    # Training
    if training_args.do_train:
        train_result = trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)

        if training_args.trainer == "hessian":
            # Write the result to log
            with FileLock('log_hessian.lock'):
                with open('log_hessian', 'a') as f:
                    train_result.update(vars(model_args))
                    train_result.update(vars(training_args))
                    train_result.update(vars(data_args))
                    if 'evaluation_strategy' in train_result:
                        train_result.pop('evaluation_strategy')
                    f.write(str(train_result) + '\n')
            exit()

        # Use the early stop, so do not save the model in the end (unless specify save_at_last)

        if training_args.trainer == "standard" or training_args.trainer == "linearhead":
            if training_args.save_at_last:
                trainer.save_model(training_args.output_dir)

            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)
                torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
                torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
            
            if training_args.evaluate_during_training:
                # Reload the best checkpoint (for eval)
                # model.load_state_dict(trainer.best_model_ckpt)
                # if training_args.prefix_tuning:
                #     # We can load prefix by directly using load_state_dict
                #     model.load_state_dict(torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin")))
                # else:
                #     model = model_fn.from_pretrained(training_args.output_dir)
                # if training_args.exclude_first_layers != -1:
                #     model = convert_opt_model(model, config, training_args.exclude_first_layers)
                
                # model = model.to(training_args.device)
                
                # Now we just reload this from memory instead of disk <-- much faster
                trainer.model.load_state_dict(trainer.best_model_ckpt)

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        ### Don't evaluate on mnli-mm for our purposes
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     test_datasets.append(
        #         FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", use_demo=('demo' in model_args.few_shot_type))
        #     )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)


    if trainer.is_world_process_zero():
        with FileLock('log.lock'):
            with open(training_args.log_file, 'a') as f:
                final_result.update(vars(model_args))
                final_result.update(vars(training_args))
                final_result.update(vars(data_args))
                if 'evaluation_strategy' in final_result:
                    final_result.pop('evaluation_strategy')
                f.write(str(final_result) + '\n')

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    return eval_results

if __name__ == "__main__":
    main()
