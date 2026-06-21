import torch.nn as nn



class RowColumnEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_dim = config.hidden_size
        self.num_embeddings1 = config.type_vocab_sizes[1]
        self.num_embeddings2 = config.type_vocab_sizes[2]
        self.token_type_embeddings_1 = nn.Embedding(self.num_embeddings1 , self.embedding_dim)
        self.token_type_embeddings_2 = nn.Embedding(self.num_embeddings2, self.embedding_dim)


    def forward(self, token_type_ids=None):

        embeddings  = self.token_type_embeddings_1(token_type_ids[:, :, 1])
        embeddings += self.token_type_embeddings_2(token_type_ids[:, :, 2])

        return embeddings
    




