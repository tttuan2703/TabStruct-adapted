# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .table_linearize import IndexedRowTableLinearize
from .table_truncate import CellLimitTruncate, RowDeleteTruncate
from .table_processor import TableProcessor
from transformers import AutoTokenizer


def get_default_processor(max_cell_length, max_input_length, input_token_structure):
    table_linearize_func = IndexedRowTableLinearize(input_token_structure)

    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large"),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor



def setup_table_processor(model_args, data_args):


    remov = 0
    remov = 200 if model_args.input_token_structure != "T0" else 2
    if data_args.dataset_name is not None:
        remov = remov if "wikisql" not in data_args.dataset_name else remov+100
    TABLE_PROCESSOR = get_default_processor(max_cell_length=15,
                                            max_input_length=data_args.max_source_length-remov,
                                            input_token_structure=model_args.input_token_structure)
    return TABLE_PROCESSOR