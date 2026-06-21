# code adapt from : https://github.com/microsoft/Table-Pretraining

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
import abc
from typing import Dict, List, List


class TableLinearize(abc.ABC):

    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """
    def __init__(self,input_token_structure):
        self.input_token_structure=input_token_structure
        

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + " "
        return _table_str.strip()
    

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        if self.input_token_structure=="T1":
            str_headers = self.process_header_T1(headers)

        if self.input_token_structure=="T2":
            str_headers = self.process_header_T2(headers)

        if self.input_token_structure=="T0":
            str_headers = self.process_header_T0(headers)
        return str_headers
    
    def process_row(self, row: List, row_index: int):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        if self.input_token_structure=="T1":
            str_row = self.process_row_T1(row, row_index)

        if self.input_token_structure=="T2":
            str_row = self.process_row_T2(row, row_index)

        if self.input_token_structure=="T0":
            str_row = self.process_row_T0(row, row_index)

        return str_row
    

    def process_header_T1(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "col : " + " | ".join(headers)

    
    def process_header_T2(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        new_header = "|| " + " ".join([f"col" for i in range(len(headers))])
        headers = " | ".join(str(headers))

        return new_header + "row |" +headers

    def process_header_T0(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "</s>" + " ".join(headers)
    
    def process_row_T2(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + row_str
    

    def process_row_T1(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + str(row_index) + " : " + row_str



    def process_row_T0(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " ".join(row_cell_values)
        return " " + row_str