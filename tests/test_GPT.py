import argparse
import numpy as np
import os
import pandas as pd
import scipy as sp
import sys
import torch
import torch.nn.functional as F
import warnings
import random
import collections

# CD-T Imports
import math
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import itertools
import pdb

from torch import nn

warnings.filterwarnings("ignore")

base_dir = os.path.split(os.getcwd())[0]
print(base_dir)
sys.path.append(base_dir)

from argparse import Namespace
from pyfunctions.cdt_basic import *
from pyfunctions.cdt_source_to_target import *
from pyfunctions.cdt_from_source_nodes import *
from pyfunctions.ioi_dataset import IOIDataset
from pyfunctions.wrappers import *
from pyfunctions.general import compare_same

from pyfunctions.ioi_dataset import IOIDataset
from transformer_lens import utils, HookedTransformer, ActivationCache

import os
# avoid some warning which takes an extremely long time having to do with the HF internals of TransformerLens
os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.autograd.set_grad_enabled(False)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TestGPT:
    @classmethod
    def setup_class(cls):
        cls.model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=False,
            refactor_factored_attn_matrices=True
        )
        cls.model.cfg.default_prepend_bos = False
        
        cls.source_list = [(Node(5, 0, 0),)]
        cls.target_nodes = [Node(7, 0, 1)]

        cls.ioi_dataset = IOIDataset(
            prompt_type="mixed", 
            N=3, 
            tokenizer=cls.model.tokenizer, 
            prepend_bos=False, 
            nb_templates=2
        )

        cls.abc_dataset = (
            cls.ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
            .gen_flipped_prompts(("S", "RAND"))
            .gen_flipped_prompts(("S1", "RAND"))
        )
        
        # Run the model on the entire dataset along the batch dimension
        abc_logits, abc_cache = cls.model.run_with_cache(cls.abc_dataset.toks)
        attention_outputs = [abc_cache['blocks.' + str(i) + '.hook_attn_out'] for i in range(12)]
        attention_outputs = torch.stack(attention_outputs, dim=1)  # now batch, head, seq, d_model
        cls.mean_acts = torch.mean(attention_outputs, dim=0)
        
        cls.logits, cls.cache = cls.model.run_with_cache(cls.ioi_dataset.toks[0])
        
        text = cls.ioi_dataset.sentences[0]
        encoding = cls.model.tokenizer.encode_plus(text, 
                                 add_special_tokens=True, 
                                 max_length=512,
                                 truncation=True, 
                                 padding = "longest", 
                                 return_attention_mask=True, 
                                 return_tensors="pt").to(device)
        encoding_idxs, attention_mask = encoding.input_ids, encoding.attention_mask
        input_shape = encoding_idxs.size()
        extended_attention_mask = get_extended_attention_mask(
            attention_mask, 
            input_shape,
            cls.model,
            device
        )
        
        cls.out_decomps, cls.target_decomps, _ = prop_GPT(
            encoding_idxs,
            extended_attention_mask, 
            cls.model,
            cls.source_list, 
            cls.target_nodes, 
            device=device, 
            mean_acts=None, 
            set_irrel_to_mean=False
        )
        
    
    def test_correctness_prop(self):
        prop_logits = cls.out_decomps[0].irrel + cls.out_decomps[0].rel
        a = compare_same(cls.logits, prop_logits)
        print(a)
        assert(compare_same(cls.logits, prop_logits))

    def test_cached_run(self):
        pre_layer_activations = [cache['blocks.' + str(i) + '.hook_resid_pre'] for i in range(12)]
        _, short_target_decomps, _ = prop_GPT(
            encoding_idxs,
            extended_attention_mask, 
            cls.model,
            cls.source_list, 
            cls.target_nodes, 
            device=device, 
            mean_acts=None, 
            set_irrel_to_mean=False,
            cached_pre_layer_acts=pre_layer_activations
        )
        assert(len(short_target_decomps) == len(cls.target_decomps))
        for i in range(len(short_target_decomps)):
            short_decomp = short_target_decomps[i]
            normal_decomp = cls.target_decomps[i]
            assert short_decomp.ablation_set == normal_decomp.ablation_set
            for j in range(len(short_decomp.target_nodes)):
                assert(short_decomp.target_nodes[j] == normal_decomp.target_nodes[j])
                assert(compare_same(short_decomp.rels[j], normal_decomp.rels[j]) > 0.99)
                assert(compare_same(short_decomp.irrels[j], normal_decomp.irrels[j]) > 0.99)
                
                
