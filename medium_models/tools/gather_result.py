import argparse
import json
import numpy as np
import torch
from torch import device
import itertools 

from transformers.optimization import get_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    parser.add_argument("--args_to_care", default=None, type=str, help="A list of args to care about. If provided, outputs which configs were not tested in your grid search")

    # These options should be kept as their default values
    parser.add_argument("--log", type=str, default="log", help="Log path.")
    parser.add_argument("--key", type=str, default='', help="Validation metric name")
    parser.add_argument("--test_key", type=str, default="", help="Test metric name")
    parser.add_argument("--test_key2", type=str, default="", help="Second test metric name")

    args = parser.parse_args()

    condition = eval(args.condition)
    if args.args_to_care is not None:
        args_to_care = eval(args.args_to_care)

    if len(args.key) == 0:
        if condition['task_name'] == 'cola':
            args.key = 'cola_dev_eval_mcc'
            args.test_key = 'cola_test_eval_mcc'
        elif condition['task_name'] == 'mrpc/acc':
            args.key = 'mrpc_dev_eval_acc'
            args.test_key = 'mrpc_test_eval_acc'
            args.test_key2 = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'mrpc/f1':
            args.key = 'mrpc_dev_eval_f1'
            args.test_key2 = 'mrpc_test_eval_acc'
            args.test_key = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'qqp/acc':
            args.key = 'qqp_dev_eval_acc'
            args.test_key = 'qqp_test_eval_acc'
            args.test_key2 = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'qqp/f1':
            args.key = 'qqp_dev_eval_f1'
            args.test_key2 = 'qqp_test_eval_acc'
            args.test_key = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'sts-b/pearson':
            args.key = 'sts-b_dev_eval_pearson'
            args.test_key = 'sts-b_test_eval_pearson'
            args.test_key2 = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'sts-b/spearmanr':
            args.key = 'sts-b_dev_eval_spearmanr'
            args.test_key2 = 'sts-b_test_eval_pearson'
            args.test_key = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'qnli':
            args.key = 'qnli_dev_eval_acc'
            args.test_key = 'qnli_test_eval_acc'
        elif condition['task_name'] == 'sst-2':
            args.key = 'sst-2_dev_eval_acc'
            args.test_key = 'sst-2_test_eval_acc'
        elif condition['task_name'] == 'snli':
            args.key = 'snli_dev_eval_acc'
            args.test_key = 'snli_test_eval_acc'
        elif condition['task_name'] == 'mnli':
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli_test_eval_mnli/acc'
        elif condition['task_name'] == 'mnli-mm':
            condition['task_name'] = 'mnli'
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli-mm_test_eval_mnli-mm/acc'
        elif condition['task_name'] == 'rte':
            args.key = 'rte_dev_eval_acc'
            args.test_key = 'rte_test_eval_acc'
        elif condition['task_name'] == 'ag_news':
            args.key = 'ag_news_dev_eval_acc'
            args.test_key = 'ag_news_test_eval_acc'
        elif condition['task_name'] == 'yahoo_answers':
            args.key = 'yahoo_answers_dev_eval_acc'
            args.test_key = 'yahoo_answers_test_eval_acc'
        elif condition['task_name'] == 'yelp_review_full':
            args.key = 'yelp_review_full_dev_eval_acc'
            args.test_key = 'yelp_review_full_test_eval_acc'
        elif condition['task_name'] == 'mr':
            args.key = 'mr_dev_eval_acc'
            args.test_key = 'mr_test_eval_acc'
        elif condition['task_name'] == 'sst-5':
            args.key = 'sst-5_dev_eval_acc'
            args.test_key = 'sst-5_test_eval_acc'
        elif condition['task_name'] == 'subj':
            args.key = 'subj_dev_eval_acc'
            args.test_key = 'subj_test_eval_acc'
        elif condition['task_name'] == 'trec':
            args.key = 'trec_dev_eval_acc'
            args.test_key = 'trec_test_eval_acc'
        elif condition['task_name'] == 'cr':
            args.key = 'cr_dev_eval_acc'
            args.test_key = 'cr_test_eval_acc'
        elif condition['task_name'] == 'mpqa':
            args.key = 'mpqa_dev_eval_acc'
            args.test_key = 'mpqa_test_eval_acc'
        else:
            raise NotImplementedError

    with open(args.log) as f:
        result_list = []
        for line in f:
            line = line.replace("<", "\"")
            line = line.replace(">", "\"")
            line = line.replace(" inf,", "float('inf'),")
            line = line.replace(" nan,", "float('nan'),")
            result_list.append(eval(line))

    seed_result = {}

    for item in result_list:
        ok = True
        for cond in condition:
            if isinstance(condition[cond], list):
                if cond not in item or (item[cond] not in condition[cond]):
                    ok = False
                    break
            else:
                if cond not in item or (item[cond] != condition[cond]):
                    ok = False
                    break
        if ok and args.test_key in item and args.key in item:
            seed = item['data_dir'].split('-')[-1] + '-' + str(item['seed'])
            if seed not in seed_result:
                seed_result[seed] = [item]
            else:
                seed_result[seed].append(item)

    all_seed_result = seed_result
    all_tags = sorted(set(x['tag'] for x in sum(all_seed_result.values(), [])))
    all_k = sorted(set(x['num_k'] for x in sum(all_seed_result.values(), [])))

    for tag in all_tags:
        for k in all_k:
            print("Tag: {}, K: {}".format(tag, k))
            seed_result_with_duplicates = {
                s: list(x for x in v if x['tag'] == tag and x['num_k'] == k)
                for s, v in all_seed_result.items()
            }
            seed_result = {
                s: list({x['output_dir']: x for x in v}.values())
                for s, v in seed_result_with_duplicates.items()
            }
            seed_best = {
                k: max(sorted(v, key=lambda x: x['output_dir']), key=lambda x: x[args.key])
                for k, v in seed_result.items() if v
            }

            ### check if all possible configs were run or not
            if args.args_to_care is not None:
                unique_arg_values = {}
                for _arg in args_to_care:
                    unique_arg_values[_arg] = []

                # collect all desired configs and all that were run 
                arg_configs = {} # all configs that were run
                for seed in seed_result.keys():
                    seed_configs = []
                    for config in seed_result[seed]:
                        if config['tag'] == tag and config['num_k'] == k:
                            _config = []
                            for _arg in args_to_care:
                                _value = config[_arg]
                                if _value not in unique_arg_values[_arg]:
                                    unique_arg_values[_arg].append(_value)
                                _config.append(_value)
                            seed_configs.append(tuple(_config))
                    arg_configs[seed] = seed_configs

                # compare to all possible configs
                missing_configs = {'seeds': []}
                for arg in args_to_care:
                    missing_configs[f'{arg}s'] = [] 

                num_missing = 0
                for seed in seed_result.keys():
                    all_possible_configs = itertools.product(*list(unique_arg_values.values()))
                    print(f'Missing configs for seed {seed}')
                    for config in all_possible_configs:
                        if config not in arg_configs[seed]:
                            missing_configs['seeds'].append(seed.split('-')[-1])
                            print(f'\t', end='')
                            for _arg, _val in zip(args_to_care, config):
                                print(f'{_arg}: {_val}', end=' ')
                                missing_configs[f'{_arg}s'].append(str(_val))
                            print()
                            num_missing += 1

                print(f'Rerun {num_missing} configs')
                for key, missed in missing_configs.items():
                    missed_str = ' '.join(missed)
                    print(f'{key}=({missed_str})')

            final_result_dev = np.zeros((len(seed_best)))
            final_result_test = np.zeros((len(seed_best)))
            final_result_test2 = np.zeros((len(seed_best)))
            final_result_loss = np.zeros((len(seed_best)))
            num_results = np.zeros((len(seed_best)))
            for i, seed in enumerate(seed_best):
                # for res in seed_result[seed]:
                #     print(res)

                final_result_dev[i] = seed_best[seed][args.key]
                final_result_test[i] = seed_best[seed][args.test_key]
                test_loss_key = args.test_key.split("_eval")[0] + "_eval_loss"
                dev_loss_key = test_loss_key.replace("test", "dev") 
                final_result_loss[i] = seed_best[seed][test_loss_key]
                num_results[i] = len(seed_result[seed])
                if len(args.test_key2) > 0:
                    final_result_test2[i] = seed_best[seed][args.test_key2]
                print("%s: best dev (%.5f, loss=%.4f) test (%.5f, loss=%.4f) %s | total trials: %d (ignored %d)" % (
                    seed,
                    seed_best[seed][args.key],
                    seed_best[seed][dev_loss_key],
                    seed_best[seed][args.test_key],
                    seed_best[seed][test_loss_key],
                    "test2 (%.5f)" % (seed_best[seed][args.test_key2]) if len(args.test_key2) > 0 else "",
                    len(seed_result[seed]),
                    len(seed_result_with_duplicates[seed]) - len(seed_result[seed])
                ))
                s = ''
                if args.args_to_care is None:
                    hp_to_care_about = [
                        'per_device_train_batch_size',
                        'gradient_accumulation_steps',
                        'learning_rate',
                        'zero_order_eps',
                        'zero_order_sample',
                        'zero_order_sample_scheduler',
                        'scale_lr_with_samples',
                        'lr_scheduler_type',
                        'weight_decay',
                    ]
                else:
                    hp_to_care_about = args_to_care
                for k in hp_to_care_about:
                    s += '| {}: {} '.format(k, seed_best[seed].get(k, ""))
                print('    ' + s)
            s = "mean +- std: "
            if len(final_result_test) > 0:
                s += "%.1f (%.1f) (#seeds %s) (#runs %s) (median %.1f)" % (final_result_test.mean() * 100, final_result_test.std() * 100, len(final_result_test), num_results.sum(), np.median(final_result_test) * 100,)
                if len(args.test_key2) > 0:
                    s += " | second metric: %.1f (%.1f) (median %.1f)" % (final_result_test2.mean() * 100, final_result_test2.std() * 100, np.median(final_result_test2) * 100)
                s += " | loss: %.4f (%.4f)" % (final_result_loss.mean(), final_result_loss.std())
            print(s)
            print("")

if __name__ == '__main__':
    main()
