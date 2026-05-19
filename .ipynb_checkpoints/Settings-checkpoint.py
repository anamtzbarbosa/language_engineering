import torch
from torch import nn, optim
from dataclasses import dataclass, asdict

# ============= Hyper-parameters for training ============== #
@dataclass

class Config :
    vocab_size: int = 5000 + 1  # This number should agree with the tokenizer
    mask_token_id: int = 5000  # new token forn unmasking
    number_of_transformer_blocks: int = 4
    number_of_attention_heads: int = 4
    vector_dim: int = 256
    block_size: int = 512
    dropout_prob: float = 0.1
    batch_size: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 0.000001
    no_of_epochs: int = 1


class TinyStoriesLM(nn.Module):

    def __init__(self, config):
        super(TinyStoriesLM, self).__init__()
        self.config = config
        self.embed =  nn.Embedding(config.vocab_size, config.vector_dim)
        self.positional = nn.Parameter(torch.randn(1, config.block_size, config.vector_dim)) # learns during trianing, beginning random
        modules = [Block(config.vector_dim,\
                         config.number_of_attention_heads,\
                         config.block_size,\
                         config.dropout_prob) for _ in range(config.number_of_transformer_blocks)]
        self.transformers = nn.ModuleList(modules)
        self.final = nn.Linear(config.vector_dim, config.vocab_size)

    def forward(self, x):
        # x size (B, S)
        # each element is a token-word -> transf it to vector
        B, S = x.shape

        # Embedding (B, S, vector_dim)
        token_embeddings = self.embed(x) # each num.token is now a 256 len vector

        # Position Embedding - size ( 1, blocksize, vector_dim) -> S increasing as model generates more words
        pos_embeddings = self.positional[:, :S, :] # (1, S, vector_dim)

        x = token_embeddings + pos_embeddings # (B, S, vector_dim)

        # TRANSFORMERS BLOCK
        for block in self.transformers:
            x = block(x)
        # output - same len but with more context

        # Project to 5_001 words. For each pos - how probable for next word to be ... from 0-4999
        logits = self.final(x) # (B, S, vector_dim) to (B, S, vocab_size)

        # YOUR CODE HERE

        return logits

    @classmethod
    def load(cls, checkpoint_path, device='cpu'):
        """
        Loads a model from a checkpoint file.
        Automatically reconstructs the config and model architecture.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = Config(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from {checkpoint_path} (Epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']})")
        return model
