from fancy_einsum import einsum

'''
These wrapper classes are used to make the GPT modules work with code intended for a HuggingFace
BERT model.
'''

class GPTLayerNormWrapper():
    
    def __init__(self, ln_module):
        self.ln_module = ln_module

    @property
    def weight(self):
        return self.ln_module.w

    @property
    def bias(self):
        return self.ln_module.b

    @property
    def eps(self):
        # this doesn't work due to a discrepancy between apparently the actual 
        # implementation of LayerNorm and the one in Clean_Transformer_Demo
        # return self.ln_module.cfg.layer_norm_eps
        return 1e-8
    
class GPTValueMatrixWrapper():
    def __init__(self, weight, bias):
        # squeeze a dimension because TLens has its value matrix separate per attention head (num_heads, d_model, d_value_rank),
        # but other code assumes a concatenated value matrix (d_value_rank, num_heads * d_model)
        # transpose because TLens multiplies on the right, but this code assumes on the left
        weight = weight.transpose(-1, -2)
        old_shape = weight.size()
        new_shape = (old_shape[0] * old_shape[1],) + (old_shape[2],)
        self.weight = (weight.reshape(new_shape)) # due to the indexing conventions, this has to reallocate memory; this may slow things down significantly
        new_bias_shape = new_shape[:-1]
        self.bias = bias.view(new_bias_shape)

class GPTOutputMatrixWrapper():
    def __init__(self, weight, bias):
        # analogous to the value matrix wrapper.
        # output matrix is separate per attention head (num_heads, d_value_rank, d_model)
        # other code assumes a concatenated value matrix (d_model, num_heads * d_value_rank)
        old_shape = weight.size()
        new_shape = (old_shape[0] * old_shape[1],) + (old_shape[2],)
        self.weight = (weight.view(new_shape))
        self.weight = self.weight.transpose(0, 1)

        new_bias_shape = new_shape[:-1]
        self.bias = bias.view(new_bias_shape)




# TODO: ensure that these conventions match the ones used by BERT on some level. We may be transposed, since TransformerLens multiplies on the right.
# NOTE: Since we don't do decomposition of attention patterns, maybe this would have been better done by just replacing the entire attention pattern calculation with a method of this attention module. 
class GPTAttentionWrapper():
    def __init__(self, attn_module):
        self.attn_module = attn_module

    def query(self, embedding):
        return einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", embedding, self.attn_module.W_Q) + self.attn_module.b_Q

    def key(self, embedding):
        return einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", embedding, self.attn_module.W_K) + self.attn_module.b_K
    

    @property
    def num_attention_heads(self):
        return self.attn_module.cfg.n_heads
    
    @property
    def attention_head_size(self):
        return self.attn_module.cfg.d_head
    
    @property
    def value(self):
        return GPTValueMatrixWrapper(self.attn_module.W_V, self.attn_module.b_V)
    
    @property
    def output(self):
        return GPTOutputMatrixWrapper(self.attn_module.W_O, self.attn_module.b_O)
    
    @property
    def all_head_size(self):
        return self.num_attention_heads * self.attention_head_size
    

'''
Helper classes for readability.
'''

class OutputDecomposition:
    def __init__(self, rel, irrel):
        self.rel = rel
        self.irrel = irrel