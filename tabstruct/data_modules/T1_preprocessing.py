import random
from tabstruct.data_modules.preprocessing_utils import get_labels
import pandas as pd
import copy


def pad_sequence_T1(sequence, max_len, pad_token_id, eos_token_id, is_token_type=False, is_attention_mask=False, logger=None):


    if is_attention_mask:
        padding_value = 0
        eos_value = 1 # attention on last token
    elif is_token_type:
        padding_value = [pad_token_id, pad_token_id, pad_token_id]
        eos_value = [0,0,0] 
    else:
        padding_value = pad_token_id
        eos_value = eos_token_id # end ofsentence
        
    if len(sequence) > max_len:
        
        if logger is not None:
            logger.info(f"Truncating sequence from {len(sequence)} to {max_len} tokens.")        
        sequence = sequence[:max_len-1]
        sequence.append(eos_value)

    return sequence + [padding_value] * (max_len - len(sequence))

def padd_all_T1(input_ids, attention_mask, token_type, tokenizer, data_args, logger=None ):

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    token_type = pad_sequence_T1(
        token_type, data_args.max_source_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_token_type=True, logger=logger
    )
    input_ids = pad_sequence_T1(
        input_ids, data_args.max_source_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, logger=logger
    )
    attention_mask = pad_sequence_T1(
        attention_mask, data_args.max_source_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_attention_mask=True, logger=logger
        )
    return input_ids, attention_mask, token_type

def flatten_table_with_query_T1(df: pd.DataFrame, query: str) -> str:
    # Create the header row
    header = f"{query} col : " + " | ".join(df.columns)

    row_ids_possible_set = list(range(3, 19))
    random.shuffle(row_ids_possible_set)
    
    # Create the rows
    rows = []
    for index, row in df.iterrows():
        row_id = index + 1
        row_str = "row {} : ".format(row_id) + " | ".join(map(str, row.values))
        rows.append(row_str)
    
    # Combine header and rows with newlines
    output = header  +" "+ " ".join(rows)
    
    return output.lower()

def add_special_tokens_T1(data_dict):
    # Extract header and rows from the input dictionary
    header = data_dict['header']
    rows = data_dict['rows']

    # Modify the header: prepend ':' to the first column header, '|' to the rest, and lower everything
    new_header = [f": {header[0].lower()}"] + [f"| {col.lower()}" for col in header[1:]]

    # Modify the rows: prepend ':' to the first column values, '|' to the rest, and lower everything
    new_rows = []
    for row in rows:
        new_row = [f": {row[0].lower()}"] + [f"| {cell.lower()}" for cell in row[1:]]
        new_rows.append(new_row)

    # Add an additional column 'col' with row labels, also lower the 'col' label
    col_labels = [f"row {i + 1}" for i in range(len(new_rows))]
    new_header = ["col"] + new_header
    new_rows = [[col_labels[i]] + new_rows[i] for i in range(len(new_rows))]

    # Return the modified dictionary
    return {'header': new_header, 'rows': new_rows}




def preprocess_tableqa_function_T1(examples, tokenizer, data_args, padding, table_processor, is_training=False, logger=None):

    pad_token_id = tokenizer.pad_token_id


    # Prepare output lists for batched inputs
    token_types, input_ids_list, attention_masks, labels_list = [], [], [], []

    # Iterate over the batch
    for example_table, query, answers in zip(examples["table"], examples["question"], examples["answers"]):
       
        query = query.lower()

        if is_training:
            example_table = table_processor.process_input(example_table, query, answers)


        # Process table
        table = add_special_tokens_T1(example_table)
        input_ids, attention_mask, token_type = get_token_type_T1(table, query, tokenizer)

        input_ids, attention_mask, token_type = padd_all_T1(input_ids, attention_mask, token_type, tokenizer, data_args, logger=None )


        labels =   get_labels(answers, data_args, tokenizer, padding, pad_token_id)
       
        token_types.append(token_type)
        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        labels_list.append(labels)

    

    return {
        "token_type": token_types,
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels_list,
    }




def get_token_type_T1(table, query, tokenizer):
    table = [table["header"]] + table["rows"]

    query_ids = tokenizer(query).input_ids[:-1] 
    query_length = len(query_ids)

    tokens_per_cells = [tokenizer([f" {I}" for I in row]).input_ids for row in table]

    rows_ids = [0] * query_length
    cols_ids = [0] * query_length
    position_ids = [0] * query_length

    input_ids = query_ids
    attention_mask = [1] *( query_length+1)

    for row_id, row in enumerate(tokens_per_cells):
        for col_id, cells in enumerate(row):
            cells = cells[1:-1]
            n_cells = len(cells)
            col_id += 1

            cols_ids.extend([col_id] * n_cells)
            rows_ids.extend([row_id] * n_cells)
            position_ids.extend([1] * n_cells)

            input_ids+= cells
            attention_mask+=[1]*( n_cells)

    input_ids+=[2] #EOS
    token_type = [[a, b, c] for a, b, c in zip(position_ids, cols_ids, rows_ids)]
    #token_type.append([0, 0, 0])

    token_type.append(token_type[-1])

    return input_ids, attention_mask, token_type


