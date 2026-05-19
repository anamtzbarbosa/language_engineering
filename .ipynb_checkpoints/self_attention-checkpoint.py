import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, vector_dim, block_size, is_causal=False):
        super().__init__()
        self.vector_dim = vector_dim
        self.is_causal = is_causal
        self.wq = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wk = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wv = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wo = nn.Linear(vector_dim, vector_dim, bias=False)
        if self.is_causal:
            # The 'causal mask' is a lower-left triangular matrix of 1s, wrapped in
            # the outermost dimension(s) (which is the batch and, in the multihead
            # case, the number_of_heads dimension)
            causal_mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0)
            # The next line creates a buffer 'self.mask'. Using a buffer rather than
            # a parameter ensures it will be moved to the GPU along with parameters,
            # but it won't be changed during training.
            self.register_buffer("mask", causal_mask)


    def compute_attention(self, q, k, v):
        # Shape of the tensors are (B,S,D) = (batch size, seq length, attention dim)
        # In the single-head case, attention_dim = vector_dim

        #dot product q • k_transpose: we use transpose so each token compares similarity/affinity (calculating dot product) with its corresponding token
        dot_prod = q @ k.transpose(-2,-1)

        #handling less extreme numbers, this helps to escale the product results in order to allow softmax to identify not only sintactic meaning
        # a extreme number would get a high probability in softmax, and interfer with the other probabilities, we would only get obvious connection between tokens
        dot_prod = dot_prod/math.sqrt(self.vector_dim)


        # We apply causal mask to dot producte matrix: because we don't want words to be influenced by the next one
        if self.is_causal:
            #causal mask size is that of block_size (max number of token), so we have to adjuste it if the number of tokens is lower
            S = dot_prod.size(-1)
            adjusted_mask = self.mask[:, :S, :S]
            dot_prod= dot_prod.masked_fill(adjusted_mask == 0, -math.inf) 

        #applying softmax (Takes dot products (affinity scores) and transforms it into probabilities, this way the models knows where to put more attention for each token)
        soft_result =torch.nn.functional.softmax(dot_prod, dim=-1)

        #computing weighted sum of values to get contextualized versions of the input tokens. y = %1 * value1 + %2 * value2 + %3 * value3 + %4 * value4
        values = soft_result @ v

        return values


    
    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        values = self.compute_attention(q, k, v)
        out = self.wo(values)
        return out



class MultiHeadSelfAttention(SelfAttention):
    def __init__(self, vector_dim, n_heads, block_size, is_causal=False):
        super().__init__(vector_dim, block_size, is_causal)
        self.att_dim = vector_dim//n_heads
        self.n_heads = n_heads


    def reshape_for_multihead_attention(self, x):
        """
        x has the shape (batch_size, seq_length, vector_dim)

        We want to split the representation of each token into 'number_of_heads'
        parts and treat each part separately. Thus, we need the returned tensor
        to have shape (batch_size, no_of_heads, seq_length, att_dim)
        """
        #Get variables from x
        batch_size, seq_length, vector_dim = x.shape

        #reshape x to get tensor with shape (batch_size, no_of_heads, seq_length, att_dim)
        x = x.reshape(batch_size, seq_length, self.n_heads, self.att_dim).transpose(1, 2)
        return x


    def reshape_after_multihead_attention(self, x):
        """
        x has the shape (batch_size, no_of_heads, seq_length, att_dim)

        For each token, we now want to bring together the representation coming
        from each head. The returned token should have the shape:
        (batch_size, seq_length, vector_dim)
        """
        #Get variables from x
        batch_size, n_heads, seq_length, att_dim = x.shape

        #reverting transpose (order)
        x = x.transpose(1, 2).contiguous()

        #reshaping to go back to (batch_size, seq_length, vector_dim)
        x = x.reshape(batch_size, seq_length, n_heads * att_dim)
        # YOUR CODE HERE

        return x



    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.reshape_for_multihead_attention(q)
        k = self.reshape_for_multihead_attention(k)
        v = self.reshape_for_multihead_attention(v)
        values = self.compute_attention(q, k, v)
        values = self.reshape_after_multihead_attention(values)
        out = self.wo(values)
        return out
