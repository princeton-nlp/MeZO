import numpy as np
import collections
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_metric(predictions, metric_name):
    if metric_name == "accuracy":
        if isinstance(predictions[0].correct_candidate, list):
            return np.mean([pred.predicted_candidate in pred.correct_candidate for pred in predictions])
        else:
            return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])
    elif metric_name == "em":
        # For question answering
        return np.mean([any([normalize_answer(ans) == normalize_answer(pred.predicted_candidate) for ans in pred.correct_candidate]) for pred in predictions])
    elif metric_name == "f1":
        # For question answering
        f1 = []
        for pred in predictions:
            all_f1s = []
            if pred.correct_candidate[0] == "CANNOTANSWER" or pred.correct_candidate[0] == "no answer":
                f1.append(int(normalize_answer(pred.correct_candidate[0]) == normalize_answer(pred.predicted_candidate)))
            else:
                for ans in pred.correct_candidate:
                    prediction_tokens = normalize_answer(pred.predicted_candidate).split()
                    ground_truth_tokens = normalize_answer(ans).split()
                    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                    num_same = sum(common.values())
                    if num_same == 0:
                        all_f1s.append(0)
                    else:
                        precision = 1.0 * num_same / len(prediction_tokens)
                        recall = 1.0 * num_same / len(ground_truth_tokens)
                        all_f1s.append((2 * precision * recall) / (precision + recall))
                f1.append(max(all_f1s))

        return np.mean(f1)


def f1(pred, gold):
    """
    This separate F1 function is used as non-differentiable metric for SQuAD
    """
    if gold[0] == "CANNOTANSWER" or gold[0] == "no answer":
        return int(normalize_answer(gold[0]) == normalize_answer(pred))
    else:
        all_f1s = []
        for ans in gold:
            prediction_tokens = normalize_answer(pred).split()
            ground_truth_tokens = normalize_answer(ans).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                all_f1s.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                all_f1s.append((2 * precision * recall) / (precision + recall))
        return np.max(all_f1s)