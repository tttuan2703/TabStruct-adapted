import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from tabstruct.bin.main_train import main_train
from tabstruct.bin.main_test import main_test
from tabstruct.utils.args import  ModelArguments, DataTrainingArguments

from transformers import HfArgumentParser
from transformers import HfArgumentParser, Seq2SeqTrainingArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.task=="train":
        main_train(model_args, data_args, training_args)
    if model_args.task=="test":
        main_test(model_args, data_args, training_args)


if __name__ == "__main__":
    main()