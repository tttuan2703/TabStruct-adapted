

from tabstruct.embeddings.CPE import CellsPositionalEmbedding
from tabstruct.embeddings.TPE import TablePositionalEmbedding
from tabstruct.embeddings.E1 import RowColumnEmbeddings
from tabstruct.embeddings.segment_embedding import SegmentEmbedding

import torch.nn as nn


class EmbeddingController(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.segment_embedding = SegmentEmbedding(config)

        # Positional Embedding (TPE, CPE)
        if config.positional_embedding == "TPE":
            self.embed_positions = TablePositionalEmbedding(config)
        elif config.positional_embedding == "CPE":
            self.embed_positions = CellsPositionalEmbedding(config)

        # Tabular Structure Embedding (E1, E0)
        self.is_structure_embedding = True if config.tabular_structure_embedding == "E1" else False
        self.tabular_structure_embedding = None
        if config.tabular_structure_embedding == "E1":
            self.tabular_structure_embedding = RowColumnEmbeddings(config)
    

    def forward(self, input_ids, token_type_ids):

        embeddings = self.segment_embedding(token_type_ids)                
        embeddings += self.embed_positions(input_ids, token_type_ids)
        if self.is_structure_embedding:
            embeddings += self.tabular_structure_embedding(token_type_ids)

        return embeddings
    
