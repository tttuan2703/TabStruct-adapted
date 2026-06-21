

import torch
from torch import nn

class SegmentEmbedding(nn.Embedding):
    """
    This module learns segment embeddings based on the provided segment ids.
    """

    def __init__(self, config):
        
        self.config = config
        # Number of segment embeddings is defined by config.type_vocab_sizes[0]
        num_embeddings = self.config.type_vocab_sizes[0]
        embedding_dim = self.config.hidden_size
        super().__init__(num_embeddings, embedding_dim)
        
    def forward(self, token_type: torch.Tensor):
        """
        Forward pass to get the segment embeddings.
        
        Args:
            token_type (torch.Tensor): The token type IDs (segment IDs), shape (batch_size, seq_len).
        
        Returns:
            torch.Tensor: The segment embeddings corresponding to the segment IDs.
        """
        # segment_ids is derived from token_type
        segment_ids = token_type[:, :, 0]
        return super().forward(segment_ids)
    
    