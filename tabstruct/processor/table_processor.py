# code adapt from : https://github.com/microsoft/Table-Pretraining

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List
from .table_linearize import TableLinearize
from .table_truncate import TableTruncate


class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 target_delimiter: str = ", "):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.target_delimiter = target_delimiter

    def process_input(self, table_content: Dict, question: str, answer: List[str]) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """

        for truncate_func in self.table_truncate_funcs:
            truncate_func.truncate_table(table_content, question, answer)


        return table_content
    
    def process_output(self, answer: List[str]) -> str:
        """
        Flatten the output for translation
        """
        output = self.target_delimiter.join(answer)
        if output.strip() == "":
            raise Exception("The Answer is EMPTY!")
        else:
            return output
