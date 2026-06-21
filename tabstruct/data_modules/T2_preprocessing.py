from tabstruct.data_modules.preprocessing_utils import get_labels
import pandas as pd
import copy

def pad_sequence_T2(sequence, max_len, pad_token_id, eos_token_id, is_token_type=False, is_attention_mask=False, logger=None):


    if is_attention_mask:
        padding_value = 0
        eos_value = 1
    elif is_token_type:
        padding_value = [pad_token_id, pad_token_id, pad_token_id, pad_token_id]
        eos_value = [0,0,0,5]
    else:
        padding_value = pad_token_id
        eos_value = eos_token_id


    if len(sequence) > max_len:
        
        if logger is not None:
            logger.info(f"Truncating sequence from {len(sequence)} to {max_len} tokens.")
        sequence = sequence[:max_len-1]
        sequence.append(eos_value)

    return sequence + [padding_value] * (max_len - len(sequence))

def padd_all_T2(input_ids, attention_mask, token_type, tokenizer, data_args, logger=None ):

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    token_type = pad_sequence_T2(
        token_type, data_args.max_source_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_token_type=True, logger=logger
    )
    input_ids = pad_sequence_T2(
        input_ids, data_args.max_source_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, logger=logger
    )
    attention_mask = pad_sequence_T2(
        attention_mask, data_args.max_source_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_attention_mask=True, logger=logger
        )
    return input_ids, attention_mask, token_type

def add_special_tokens_T2(data_dict):

    header = data_dict['header']
    rows = data_dict['rows']
    rows.insert(0, header)  # Treat the header as a row in the data
    new_header = ["||"] + ["col" for _ in header]  # Add CLS token and replace headers with 'col'
    new_rows = [["row"] + [f"| {cell.lower()}" for cell in row] for row in rows]

    return {'header': new_header, 'rows': new_rows}




def get_token_type_T2(table, query, tokenizer):
    table = [table["header"]] + table["rows"]

    query_ids = tokenizer(query).input_ids[:-1] 
    query_length = len(query_ids)


    tokens_per_cells = [tokenizer([f" {I.lower().strip()}" for I in row]).input_ids for row in table]

    rows_ids = [0] * query_length
    cols_ids = [0] * query_length
    position_ids = [0] * query_length
    special_ids = [5]+([0] * (query_length-1))

    input_ids = query_ids
    attention_mask = [1] * (query_length+1)

    for row_id, row in enumerate(tokens_per_cells):

        position_id = 1 

        for col_id, cells in enumerate(row):
            cells = cells[1:-1]
            n_cells = len(cells)
            col_id += 1
            if len(cells)>0:
                if cells[0] == 1721:
                    special_id = [4] + [0]*( n_cells-1)
                elif cells[0] == 3236:
                    special_id = [2] + [0]*( n_cells-1)
                elif cells[0] == 11311:
                    special_id = [3] + [0]*( n_cells-1)
                elif cells[0] == 45056:
                    special_id = [1] + [0]*( n_cells-1)
                    
                else : 
                    special_id = [0] * n_cells

            else : 
                special_id = [0] * n_cells
            
            special_ids.extend(special_id)
            cols_ids.extend([col_id] * n_cells)
            rows_ids.extend([row_id] * n_cells)
            position_ids.extend([position_id] * n_cells)
            input_ids+= cells
            attention_mask+=[1]*( n_cells)
            
    input_ids+=[2] #EOS
    token_type = [[a, b, c, d] for a, b, c, d in zip(position_ids, cols_ids, rows_ids, special_ids)]
    #token_type.append([0, 0, 0, 5])

    token_type.append(token_type[-1])

    return input_ids, attention_mask, token_type



def preprocess_tableqa_function_T2(examples, tokenizer, data_args, padding, table_processor, is_training=False, logger=None):

    # Prepare output lists for batched inputs
    token_types, input_ids_list, attention_masks, labels_list = [], [], [], []

    # Iterate over the batch
    for example_table, query, answers in zip(examples["table"], examples["question"], examples["answers"]):
       
        query = query.lower()


        if is_training:
            example_table = table_processor.process_input(example_table, query, answers)



        table = add_special_tokens_T2(example_table)
        input_ids, attention_mask, token_type = get_token_type_T2(table, query, tokenizer)

        input_ids, attention_mask, token_type = padd_all_T2(input_ids, attention_mask, token_type, tokenizer, data_args, logger=None )

        labels =   get_labels(answers, data_args, tokenizer, padding, tokenizer.pad_token_id)


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

