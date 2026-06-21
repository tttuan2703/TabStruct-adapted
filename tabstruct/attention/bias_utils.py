import torch 


def get_relative_relation_ids(token_type, input_mask, 
                              disabled_attention_bias = 0):
    # Adapt from : https://github.com/google-research/tapas/blob/master/tapas/utils/tableformer_utils.py


    # token_type (bsz, seq_length, 3)


    segment_ids = token_type[:,:, 0]
    column_ids = token_type[:,:, 1]
    row_ids = token_type[:,:, 2]

    input_mask = 1-input_mask.squeeze(1).sum(dim=1).bool().int()

    cell_mask = (row_ids != 0) & (segment_ids == 1) & (input_mask == 1)
    header_mask = (row_ids == 0) & (segment_ids == 1) & (input_mask == 1)
    sent_mask = (input_mask == 1) & (segment_ids == 0)
    relative_attention_ids = []
    
  # "Same row" attention bias with type id = 1.
    if disabled_attention_bias != 1:
        same_row_bias = (
            (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
            cell_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * 1
        relative_attention_ids.append(same_row_bias)
    
    else:
        column_and_row_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
            cell_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * 1
        relative_attention_ids.append(column_and_row_bias)
        

    # "Same column" attention bias matrix with type id = 2.
    if disabled_attention_bias != 2:
        same_column_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            cell_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * 2
        relative_attention_ids.append(same_column_bias)
    else:
        column_and_row_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
            cell_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * 2
        relative_attention_ids.append(column_and_row_bias)
    

    # "Same cell" attention bias matrix with type id = 3.
    if disabled_attention_bias == 3:
        same_cell_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
            cell_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * -3
        relative_attention_ids.append(same_cell_bias)
    
    # "Cell to its header" bias matrix with type id = 4.
    if disabled_attention_bias != 4:
        cell_to_header_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            header_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * 4
        relative_attention_ids.append(cell_to_header_bias)
        

    # "Cell to sentence" bias matrix with type id = 5.
    if disabled_attention_bias != 5:
        cell_to_sentence_bias = (
            sent_mask.unsqueeze(1) &
            cell_mask.unsqueeze(2)
        ).int() * 5
        relative_attention_ids.append(cell_to_sentence_bias)

    # "Header to column cell" bias matrix with type id = 6.
    if disabled_attention_bias != 6:
        header_to_column_cell_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            cell_mask.unsqueeze(1) &
            header_mask.unsqueeze(2)
        ).int() * 6
        relative_attention_ids.append(header_to_column_cell_bias)

  # "Header to other header" bias matrix with type id = 7.
    if disabled_attention_bias != 7:
        header_to_other_header_bias = (
            header_mask.unsqueeze(1) &
            header_mask.unsqueeze(2)
        ).int() * 7
        relative_attention_ids.append(header_to_other_header_bias)

    # "Header to same header" bias matrix with type id = 8.
    if disabled_attention_bias != 8:
        if disabled_attention_bias != 7:
            header_to_same_header_bias = (
                (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                header_mask.unsqueeze(1) &
                header_mask.unsqueeze(2)
            ).int() * 1
            relative_attention_ids.append(header_to_same_header_bias)
        else:
            header_to_same_header_bias = (
                (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                header_mask.unsqueeze(1) &
                header_mask.unsqueeze(2)
            ).int() * 8
            relative_attention_ids.append(header_to_same_header_bias)
    else:
        header_to_same_header_bias = (
            (column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
            (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
            header_mask.unsqueeze(1) &
            header_mask.unsqueeze(2)
        ).int() * -7
        relative_attention_ids.append(header_to_same_header_bias)
    # "Header to sentence" bias matrix with type id = 9.
    if disabled_attention_bias != 9:
        header_to_sentence_bias = (
            sent_mask.unsqueeze(1) &
            header_mask.unsqueeze(2)
        ).int() * 9
        relative_attention_ids.append(header_to_sentence_bias)

    # "Sentence to cell" bias matrix with type id = 10.
    if disabled_attention_bias != 10:
        sentence_to_cell_bias = (
            cell_mask.unsqueeze(1) &
            sent_mask.unsqueeze(2)
        ).int() * 10
        relative_attention_ids.append(sentence_to_cell_bias)

    # "Sentence to header" bias matrix with type id = 11.
    if disabled_attention_bias != 11:
        sentence_to_header_bias = (
            header_mask.unsqueeze(1) &
            sent_mask.unsqueeze(2)
        ).int() * 11
        relative_attention_ids.append(sentence_to_header_bias)

    # "Sentence to sentence" bias matrix with type id = 12.
    if disabled_attention_bias != 12:
        sentence_to_sentence_bias = (
            sent_mask.unsqueeze(1) &
            sent_mask.unsqueeze(2)
        ).int() * 12
        relative_attention_ids.append(sentence_to_sentence_bias)
    

    return torch.sum(torch.stack(relative_attention_ids), dim=0).int()

