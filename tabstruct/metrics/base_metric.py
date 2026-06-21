from collections import defaultdict
from typing import List
import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels



# define example evaluation
def evaluate_example(predict_str: str, ground_str: str):
    delimiter = ", "
    predict_spans = predict_str.split(delimiter)
    ground_spans = ground_str.split(delimiter)
    predict_values = defaultdict(lambda: 0)
    ground_values = defaultdict(lambda: 0)
    for span in predict_spans:
        try:
            predict_values[float(span)] += 1
        except ValueError:
            predict_values[span.strip()] += 1
    for span in ground_spans:
        try:
            ground_values[float(span)] += 1
        except ValueError:
            ground_values[span.strip()] += 1
    _is_correct = predict_values == ground_values
    return _is_correct

def get_denotation_accuracy(predictions: List[str], references: List[str]):
    assert len(predictions) == len(references)
    correct_num = 0
    for predict_str, ground_str in zip(predictions, references):
        is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
        if is_correct:
            correct_num += 1
    return correct_num / len(predictions)


def compute_metrics(eval_preds, tokenizer, data_args):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]


    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(f" decoded_labels : {decoded_labels}")
    print(f" decoded_preds : {decoded_preds}")
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    accuracy = get_denotation_accuracy(decoded_preds, decoded_labels)
    result = {"denotation_accuracy": accuracy}
    print(f'result : {result}')
    return result