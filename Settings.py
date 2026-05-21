import torch
from torch import nn, optim
from dataclasses import dataclass, asdict
from self_attention import MultiHeadSelfAttention

# ============= Hyper-parameters for training ============== #

class PositionwiseFFN(nn.Module):
    """
    The position-wise FFN that follows after the self-attention computation.
    Vectors are projected to 4x the dimensionality and then projected down
    again after relu application.
    """

    def __init__(self, vector_dim, dropout_prob) :
        super().__init__()
        self.fc1 = nn.Linear(vector_dim, 4*vector_dim, bias=True)
        self.fc2 = nn.Linear(4*vector_dim, vector_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class Block(nn.Module):
    """
    Transformer encoder block.

    This version differs from the original version in  [Vaswani et al. NeurIPS 2017],
    and applies the LayerNorm before the self-attention, and before the FFN, as this
    has proved to be beneficial (see [Nguyen and Salazar 2019]).
    """

    def __init__(self, vector_dim, n_heads, block_size, dropout_prob):
        super().__init__()
        att_dim = vector_dim // n_heads
        self.attn = MultiHeadSelfAttention(vector_dim, n_heads, block_size, is_causal=True)
        self.ffn = PositionwiseFFN(vector_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(vector_dim)
        self.ln2 = nn.LayerNorm(vector_dim)

    def forward(self, x):
        x1 = self.ln1(x)
        x2 = x + self.dropout(self.attn(x1))
        x3 = self.ln2(x2)
        x4 = x2 + self.dropout(self.ffn(x3))
        return x4

        
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
