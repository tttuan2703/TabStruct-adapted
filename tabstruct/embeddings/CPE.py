
from transformers.models.tapas.modeling_tapas import IndexMap, ProductIndexMap, reduce_min, gather

import torch

class CellsPositionalEmbedding(torch.nn.Embedding):
    # code adapt from : https://github.com/huggingface/transformers/blob/main/src/transformers/models/tapas/modeling_tapas.py
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, config):

        self.config = config

        self.num_embeddings = config.max_position_embeddings
        self.embedding_dim = config.hidden_size
        self.offset = 2

        super().__init__(self.num_embeddings + self.offset, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor, token_type):

        """`input_ids' shape is expected to be [bsz x seqlen]."""

        _, seq_len = input_ids.shape[:2]
        device = input_ids.device 
        input_shape = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # shape (batch_size, seq_len)
        col_index = IndexMap(token_type[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
        # shape (batch_size, seq_len)
        row_index = IndexMap(token_type[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
        # shape (batch_size, seq_len)
        full_index = ProductIndexMap(col_index, row_index)
        # shape (max_rows * max_columns,). First absolute position for every cell
        first_position_per_segment = reduce_min(position_ids, full_index)[0]
        # ? shape (batch_size, seq_len). First absolute position of the cell for every token
        first_position = gather(first_position_per_segment, full_index)
        # shape (1, seq_len)
        position = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        positions = torch.min(
            torch.as_tensor(self.config.max_position_embeddings - 1, device=device), position - first_position
        )

  

                
        return super().forward(positions + self.offset)
    



