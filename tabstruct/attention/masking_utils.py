import torch 



def gen_mask(token_type, name) :

    segment_ids = token_type[:,:, 0]
    column_ids = token_type[:,:, 1]
    row_ids = token_type[:,:, 2]

    cell_mask = (row_ids != 0) & (segment_ids == 1)
    header_mask = (row_ids == 0) & (segment_ids == 1)
    
    if name.startswith("STM"):
        special_ids = token_type[:, :, 3]
        special_token_sep_mask = (special_ids==1) & (segment_ids == 1) 
        special_token_row_mask = (special_ids==2) & (segment_ids == 1) 
        special_token_col_mask = (special_ids==3) & (segment_ids == 1) 
        special_token_cells_mask = (special_ids==4) & (segment_ids == 1) 

    if name == 'M_query': 
        segment_zero = segment_ids == 0
        mask =  segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1)
        
    if name == "M_self": 
        device = token_type.device
        seq_len = token_type.size(1)
        arange_tensor = torch.arange(1, seq_len + 1, device=device)
        mask = (arange_tensor.unsqueeze(0).unsqueeze(1) == arange_tensor.unsqueeze(0).unsqueeze(2))

    if name == "M_cells":
        mask8 = column_ids.unsqueeze(1) == column_ids.unsqueeze(2)
        mask9 = row_ids.unsqueeze(1) == row_ids.unsqueeze(2)
        mask = mask8 & mask9
        
    if name == "M_columns":
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)))
        
    if name == "M_rows":
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)))
        
    if name == "STM_1": # Special Token Row -> Row
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)))
    
    if name == "STM_2": # Special Token Cell -> Cell
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2)))
    
    if name == "STM_3": # Special Token Col -> Col
        cell_to_header_mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                              header_mask.unsqueeze(1) &
                              cell_mask.unsqueeze(2) & 
                              ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        special_col_attention = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                cell_mask.unsqueeze(1) &
                                header_mask.unsqueeze(2) &
                                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        mask = torch.logical_or(special_col_attention, cell_to_header_mask)


    if name == "STM_4": # Special Token table -> table
        mask = (~(special_token_sep_mask.unsqueeze(1) == special_token_sep_mask.unsqueeze(2))&
               (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)))

    return mask






def generate_mask(token_type, attention_mask, mask_number):
    
    attention_mask_global = (attention_mask.unsqueeze(1)==1) & ( 1 == attention_mask.unsqueeze(2)) 
        
    if mask_number==1: # M1
        mask = (  gen_mask(token_type, "M_query") |
                  gen_mask(token_type, "M_rows")  |
                  gen_mask(token_type, "M_columns") ) 
    
    elif mask_number==2: #M2
        mask = (  gen_mask(token_type, "M_query")  | 
                  gen_mask(token_type, "M_columns"))
        
    elif mask_number==3: #M3
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "M_rows"))

    elif mask_number==4: 
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self") |
                  gen_mask(token_type, "M_columns"))

    elif mask_number==5: 
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self") |
                  gen_mask(token_type, "M_rows"))

    elif mask_number==6: 
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self") |
                  gen_mask(token_type, "M_cells"))
        
    
    attention_mask = (mask & attention_mask_global) 

    inverted_mask = 1.0 - attention_mask.float()
    inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inverted_mask.dtype).min)

    return inverted_mask