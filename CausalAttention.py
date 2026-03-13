# Applying a causal attention and a dropout mask
#  At each stage the LLM is supposed to only predict the next token, hence the future tokens need to be masked.
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,seed=123,qkv_bias=False):
        super().__init__()
        torch.manual_seed(seed)
        self.W_query = torch.nn.Linear(d_in,d_out,bias=qkv_bias) #slightly better weight initialization under the hood
        self.W_key = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1)) #register_buffer helps register tensors to move over to the GPU

    def forward(self,x):
        b,num_tokens,d_in = x.shape
        queries = self.W_query(x)
        keys =  self.W_key(x)
        d_k = keys.shape[1]
        values = self.W_value(x)
        attn_scores = queries@keys.T
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf) # in pytorch the _ suffix for function is inplace convention 
        attn_weights = torch.softmax(attn_scores/d_k**0.5, dim = 1)
        attn_weights= self.dropout(attn_weights)
        context_vector = attn_weights@values
        return context_vector