

import torch

class TablePositionalEmbedding(torch.nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, config):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack

        self.num_embeddings = config.max_position_embeddings
        self.embedding_dim = config.hidden_size
        self.offset = 2

        super().__init__(self.num_embeddings + self.offset, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor, token_type ):

        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)



        return super().forward(positions + self.offset)