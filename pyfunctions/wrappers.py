from fancy_einsum import einsum
from typing import NamedTuple
import torch
from dataclasses import dataclass

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
        # analogous to the value matrix wrappeExpected size for first two dimensions of batch2 tensor to be: [768, 768] but got: [768, 12].
        new_shape = (old_shape[0] * old_shape[1],) + (old_shape[2],)
        self.weight = (weight.view(new_shape))
        self.weight = self.weight.transpose(0, 1)

        new_bias_shape = new_shape[:-1]
        self.bias = bias.view(new_bias_shape)


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

'''
class OutputDecomposition:
    def __init__(self, source_node, rel, irrel):
        self.source_node = source_node
        self.rel = rel
        self.irrel = irrel
'''
class Node(NamedTuple):
    layer_idx: int
    sequence_idx: int
    attn_head_idx: int

# "ablation" isn't the right thing to call this, exactly; it's the set of nodes
# you want to decompose into (rel, irrel) in the forward pass
# however, it performs a function analogous to ablation in other interpretability techniques,
# in that we can determine the importance of these nodes

'''
class AblationSet:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
'''
type AblationSet = tuple[Node]


class OutputDecomposition(NamedTuple):
    ablation_set: AblationSet
    rel: torch.Tensor
    irrel: torch.Tensor

@dataclass
class TargetNodeDecompositionList:
    ablation_set: AblationSet
    target_nodes: list[Node]
    rels: list[torch.Tensor]
    irrels: list[torch.Tensor]
    
    def __init__(self, ablation_set: AblationSet):
        self.ablation_set = ablation_set
        self.target_nodes = []
        self.rels = []
        self.irrels = []

    def append(self, target_node, rel, irrel):
        self.target_nodes.append(target_node)
        self.rels.append(rel)
        self.irrels.append(irrel)

    # hopefully this doesn't slow things down too much with a bunch of reallocations
    def __add__(self, other):
        assert self.ablation_set == other.ablation_set
        s = TargetNodeDecompositionList(self.ablation_set)
        s.target_nodes = self.target_nodes + other.target_nodes
        s.rels = self.rels + other.rels
        s.irrels = self.irrels + other.irrels
        return s
