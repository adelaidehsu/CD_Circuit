import numpy as np
import warnings
import torch
import math
import pdb
from torch import nn
import transformer_lens
from transformers.modeling_utils import ModuleUtilsMixin

# utility
def normalize_rel_irrel(rel, irrel):
    tot = rel + irrel
    tot_mask = (rel * irrel) < 0
    rel_mask = tot_mask & (rel.abs() >= irrel.abs())
    irrel_mask = tot_mask & (~rel_mask)
    
    rel[rel_mask] = tot[rel_mask]
    rel[irrel_mask] = 0
    
    irrel[irrel_mask] = tot[irrel_mask]
    irrel[rel_mask] = 0

def get_encoding(text, tokenizer, device, max_seq_len=512):
    encoding = tokenizer.encode_plus(text, 
                                 add_special_tokens=True, 
                                 max_length=max_seq_len,
                                 truncation=True, 
                                 padding = "max_length", 
                                 return_attention_mask=True, 
                                 pad_to_max_length=True,
                                 return_tensors="pt").to(device)
    return encoding

def get_embeddings_bert(encoding, model):
    embedding_output = model.bert.embeddings(
            input_ids=encoding['input_ids'],
            position_ids=None,
            token_type_ids=encoding['token_type_ids'],
            inputs_embeds=None,
        )
    return embedding_output

def get_att_list(embedding_output, rel_pos, 
                 extended_attention_mask, encoder_model):
    att_scores = ()
    act = embedding_output
    
    for i, layer_module in enumerate(encoder_model.layer):
        key =  layer_module.attention.self.key(act)
        query =  layer_module.attention.self.query(act)

        att_probs = get_attention_scores(query, key, 
                                         extended_attention_mask, 
                                         rel_pos, layer_module.attention.self)
        
        att_scores = att_scores + (att_probs,)
        
        act = layer_module(act, 
                           attention_mask = extended_attention_mask,
                           rel_pos = rel_pos)[0]
    
    return att_scores

# prop functions
"""
def prop_act(rel, irrel, act_module):
    rel_a = act_module(rel)
    irrel_a = act_module(irrel)
    tot_a = act_module(rel + irrel)
    rel_act = (rel_a + (tot_a - irrel_a)) / 2
    irrel_act = (irrel_a + (tot_a - rel_a)) / 2
    return rel_act, irrel_act
"""

# TODO: does this work correctly for all activations, or is this assuming GELU, or something else?
def prop_act(r, ir, act_mod):
    ir_act = act_mod(ir)
    r_act = act_mod(r + ir) - ir_act
    return r_act, ir_act
x -> Wx + B
x = rel + irrel
rel_new + irrel_new = Wx + b 
def prop_linear_core(rel, irrel, W, b, tol = 1e-8):
    rel_t = torch.matmul(rel, W)
    irrel_t = torch.matmul(irrel, W)    

    exp_bias = b.expand_as(rel_t)
    tot_wt = torch.abs(rel_t) + torch.abs(irrel_t) + tol
    
    rel_bias = exp_bias * (torch.abs(rel_t) / tot_wt)
    irrel_bias = exp_bias * (torch.abs(irrel_t) / tot_wt)
    
    # tot_pred = rel_bias + rel_t + irrel_bias + irrel_t
    
    return (rel_t + rel_bias), (irrel_t + irrel_bias)

def prop_linear(rel, irrel, linear_module):
    return prop_linear_core(rel, irrel, linear_module.weight.T, linear_module.bias)

def prop_GPT_unembed(rel, irrel, unembed_module):
    return prop_linear_core(rel, irrel, unembed_module.W_U, unembed_module.b_U)


def prop_layer_norm(rel, irrel, layer_norm_module, tol = 1e-8):
    tot = rel + irrel
    rel_mn = torch.mean(rel, dim = 2).unsqueeze(-1).expand_as(rel)
    irrel_mn = torch.mean(irrel, dim = 2).unsqueeze(-1).expand_as(irrel)
    vr = ((torch.mean(tot ** 2, dim = 2) - torch.mean(tot, dim = 2) ** 2)
          .unsqueeze(-1).expand_as(tot))
    
    rel_wt = torch.abs(rel)
    irrel_wt = torch.abs(irrel)
    tot_wt = rel_wt + irrel_wt + tol
    '''
    # huge hack; instead can refactor function signature but i don't have the tools to do this without editing in at least 30 places
    if hasattr(layer_norm_module, "eps"):
        epsilon = layer_norm_module.eps
        weight = layer_norm_module.weight
        bias = layer_norm_module.bias
    else:
        epsilon = layer_norm_module.cfg.layer_norm_eps
        weight = layer_norm_module.w
        bias = layer_norm_module.b
    '''

    rel_t = ((rel - rel_mn) / torch.sqrt(vr + layer_norm_module.eps)) * layer_norm_module.weight
    irrel_t = ((irrel - irrel_mn) / torch.sqrt(vr + layer_norm_module.eps)) * layer_norm_module.weight
    
    rel_bias = layer_norm_module.bias * (rel_wt / tot_wt)
    irrel_bias = layer_norm_module.bias * (irrel_wt / tot_wt)
    
    return rel_t + rel_bias, irrel_t + irrel_bias

def prop_pooler(rel, irrel, pooler_module):
    rel_first = rel[:, 0]
    irrel_first = irrel[:, 0]
    
    rel_lin, irrel_lin = prop_linear(rel_first, irrel_first, pooler_module.dense)
    rel_out, irrel_out = prop_act(rel_lin, irrel_lin, pooler_module.activation)
    
    return rel_out, irrel_out

def prop_classifier_model(encoding, rel_ind_list, model, device, max_seq_len, att_list = None):
    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding['input_ids'].size()
    extended_attention_mask = get_extended_attention_mask(attention_mask = encoding['attention_mask'], 
                                                          input_shape = input_shape, 
                                                          model = model.bert,
                                                         device=device)
    
    
    tot_rel = len(rel_ind_list)
    sh = list(embedding_output.shape)
    sh[0] = tot_rel
    
    rel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    irrel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    
    for i in range(tot_rel):
        rel_inds = rel_ind_list[i]
        mask = np.isin(np.arange(max_seq_len), rel_inds)

        rel[i, mask, :] = embedding_output[0, mask, :]
        irrel[i, ~mask, :] = embedding_output[0, ~mask, :]
    
    head_mask = [None] * model.bert.config.num_hidden_layers
    rel_enc, irrel_enc = prop_encoder(rel, irrel, 
                                      extended_attention_mask, 
                                      head_mask, model.bert.encoder, att_list)
    rel_pool, irrel_pool = prop_pooler(rel_enc, irrel_enc, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)
    
    return rel_out, irrel_out

# propogate code for attention modules
def transpose_for_scores(x, sa_module):
    # handle different attention calculation conventions:
    # if it's the "Standard" attention calculation, all the key and query matrices are concatenated,
    # so the current dimension is [batch, sequence_idx, attention_heads * attn_dim]
    # and we need to unroll it.
    # however, some models do this automatically
    if len(x.size()) == 3:
        new_x_shape = x.size()[:-1] + (sa_module.num_attention_heads, sa_module.attention_head_size)
        x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def mul_att(att_probs, value, sa_module):
    context_layer = torch.matmul(att_probs, transpose_for_scores(value, sa_module))
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (sa_module.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer

'''
Don't read too much into this code; it's taken from transformers.modeling_utils.py
with hacky alterations to make it work with our code.
This function may not actually be necessary, depending on what the shapes
of the inputs are.
TODO: determine whether this function is necessary or vestigial and then update this comment'''
def get_extended_attention_mask(attention_mask, input_shape, model, device):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    dtype = next(model.parameters()).dtype

    is_decoder = False
    if (hasattr(model, 'config') and model.config.is_decoder):
        is_decoder = True
    if isinstance(model, transformer_lens.HookedTransformer):
        is_decoder = True # hack; just for GPT2 model
    if not (attention_mask.dim() == 2 and is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def get_attention_probs(tot_embed, attention_mask, head_mask, sa_module):
    mixed_query_layer = sa_module.query(tot_embed) # these parentheses are the call to forward(), i think it's easiest to implement another wrapper class

    key_layer = transpose_for_scores(sa_module.key(tot_embed), sa_module)

    query_layer = transpose_for_scores(mixed_query_layer, sa_module)
    
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(sa_module.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
    
    '''
    layer_count = attention_scores.shape[1]
    W_o = o_module_dense.weight #[768, 768]
    #print(value.shape) #[1, 512, 768]
    info = torch.matmul(W_o, value.transpose(-2, -1)).transpose(-2, -1) #[1, 512, 768] - W_o*v
    info = info.reshape(info.shape[0], info.shape[1], layer_count, -1) ##[1, 512, 12, 64]
    info_norm = torch.norm(info, p=2, dim=-1) #[1, 512, 12]
    info_norm = info_norm.transpose(-1, -2) #[1, 12, 512]
    info_norm = info_norm.unsqueeze(3).repeat(1, 1, 1, info_norm.shape[-1]) #[1, 12, 512, 512]
    # Info-weighted
    info_w_attention_probs = nn.functional.softmax(attention_scores / info_norm, dim=-1) #[1, 12, 512, 512]
    '''
    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1) #[1, 12, 512, 512]
    

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    return attention_probs

def prop_self_attention(rel, irrel, attention_mask, head_mask, sa_module, att_probs = None):
    if att_probs is not None:
        att_probs = att_probs
    else:
        att_probs = get_attention_probs(rel + irrel, attention_mask, head_mask, sa_module)
    
    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)
    
    rel_context = mul_att(att_probs, rel_value, sa_module)
    irrel_context = mul_att(att_probs, irrel_value, sa_module)
    
    return rel_context, irrel_context

def prop_attention(rel, irrel, attention_mask, head_mask, a_module, att_probs = None):
    rel_context, irrel_context = prop_self_attention(rel, irrel, 
                                                     attention_mask, 
                                                     head_mask, 
                                                     a_module.self, att_probs)
    normalize_rel_irrel(rel_context, irrel_context) # add
    
    output_module = a_module.output
    
    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    normalize_rel_irrel(rel_dense, irrel_dense) # add
    
    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    normalize_rel_irrel(rel_tot, irrel_tot) # add
    
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)
    normalize_rel_irrel(rel_out, irrel_out) # add
    
    return rel_out, irrel_out

def prop_layer(rel, irrel, attention_mask, head_mask, layer_module, att_probs = None):
    rel_a, irrel_a = prop_attention(rel, irrel, attention_mask, head_mask, layer_module.attention, att_probs)
    
    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    normalize_rel_irrel(rel_id, irrel_id) # add
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)
    
    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    normalize_rel_irrel(rel_od, irrel_od) # add
    
    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    normalize_rel_irrel(rel_tot, irrel_tot) # add
    
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)
        
    return rel_out, irrel_out

def prop_encoder(rel, irrel, attention_mask, head_mask, encoder_module, att_list = None):
    rel_enc, irrel_enc = rel, irrel
    att_scores = ()
    for i, layer_module in enumerate(encoder_module.layer):
        att_probs = att_list[i] if att_list is not None else None
        layer_head_mask = head_mask[i]
        
        rel_enc_n, irrel_enc_n = prop_layer(rel_enc, irrel_enc, attention_mask, layer_head_mask, layer_module, att_probs)
        
        normalize_rel_irrel(rel_enc_n, irrel_enc_n)
        rel_enc, irrel_enc = rel_enc_n, irrel_enc_n
    
    return rel_enc, irrel_enc


def prop_encoder_from_level(rel, irrel, attention_mask, head_mask, encoder_module, level = 0, att_list = None):
    rel_enc, irrel_enc = rel, irrel
    att_scores = ()
    for i, layer_module in enumerate(encoder_module.layer):
        if i < level:
            continue
        att_probs = att_list[i] if att_list is not None else None
        layer_head_mask = head_mask[i]
        
        rel_enc_n, irrel_enc_n = prop_layer(rel_enc, irrel_enc, attention_mask, layer_head_mask, layer_module, att_probs)
        
        normalize_rel_irrel(rel_enc_n, irrel_enc_n)
        rel_enc, irrel_enc = rel_enc_n, irrel_enc_n
    
    return rel_enc, irrel_enc

def prop_classifier_model_from_level(encoding, rel_ind_list, model, device, max_seq_len, level = 0, att_list = None):
    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding['input_ids'].size()
    extended_attention_mask = get_extended_attention_mask(attention_mask = encoding['attention_mask'], 
                                                          input_shape = input_shape, 
                                                          model = model.bert,
                                                         device = device)
    
    head_mask = [None] * model.bert.config.num_hidden_layers
    encoder_module = model.bert.encoder
    
    for i, layer_module in enumerate(encoder_module.layer):
        if i == level:
            break
        embedding_output = layer_module(embedding_output, 
                                        extended_attention_mask,
                                        head_mask[i])[0]
    
    tot_rel = len(rel_ind_list)
    sh = list(embedding_output.shape)
    sh[0] = tot_rel
    
    rel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    irrel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    
    for i in range(tot_rel):
        rel_inds = rel_ind_list[i]
        mask = np.isin(np.arange(max_seq_len), rel_inds)

        rel[i, mask, :] = embedding_output[0, mask, :]
        irrel[i, ~mask, :] = embedding_output[0, ~mask, :]
    
    
    rel_enc, irrel_enc = prop_encoder_from_level(rel, irrel, 
                                                 extended_attention_mask, 
                                                 head_mask, encoder_module, level)
    rel_pool, irrel_pool = prop_pooler(rel_enc, irrel_enc, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)
    
    return rel_out, irrel_out

def comp_cd_scores_level_skip(model, encoding, label, le_dict, device, max_seq_len, level = 0, skip = 1, num_at_time = 64):

    closest_competitor, lab_index = get_closest_competitor(model, encoding, label, le_dict)
    
    L = int((encoding['input_ids'] != 0).long().sum())
    tot_rel, tot_irrel = prop_classifier_model_from_level(encoding, 
                                                          [get_rel_inds(0, L - 1)],
                                                          model,
                                                          device,
                                                          max_seq_len = max_seq_len,
                                                          level = level)
    tot_score = proc_score(tot_rel[0, :], lab_index, closest_competitor)
    tot_irrel_score = proc_score(tot_irrel[0, :], lab_index, closest_competitor)

    # get scores
    unit_rel_ind_list = [get_rel_inds(i, min(L - 1, i + skip - 1)) for i in range(0, L, skip)]

    def proc_num_at_time(ind_list):
        scores = np.empty(0)
        irrel_scores = np.empty(0) # for ablation purposes
        L = len(ind_list)
        for i in range(int(L / num_at_time) + 1):
            cur_scores, cur_irrel = prop_classifier_model_from_level(encoding, 
                                                            ind_list[i * num_at_time: min(L, (i + 1) * num_at_time)], 
                                                            model,
                                                            device,
                                                            max_seq_len = max_seq_len,
                                                            level = level)
            #cur_scores = np.array([proc_score(cur_scores[i, :], lab_index, closest_competitor) - tot_score 
            #                       for i in range(cur_scores.shape[0])])
            cur_scores = np.array([proc_score(cur_scores[i, :], lab_index, closest_competitor) for i in range(cur_scores.shape[0])])
            scores = np.append(scores, cur_scores)
            
            cur_irrel = np.array([proc_score(cur_irrel[i, :], lab_index, closest_competitor) for i in range(cur_irrel.shape[0])])
            irrel_scores = np.append(irrel_scores, cur_irrel)
        return scores, irrel_scores

    scores, irrel_scores = proc_num_at_time(unit_rel_ind_list)
    
    return scores, irrel_scores


def get_closest_competitor(model, encoding, label, le_dict):
    
    model_output = model(**encoding)
    lab_index = le_dict[label]

    output = model_output[0].clone().cpu().detach().numpy().squeeze()
    sort_inds = np.argsort(output)

    if sort_inds[-1] != lab_index:
        return sort_inds[-1], lab_index
    else:
        return sort_inds[-2], lab_index

# Custom Score Processing function
def proc_score(tot_score, lab_index, closest_competitor):
    return float(tot_score[lab_index] - tot_score[closest_competitor])

def get_rel_inds(start, stop):
    return list(range(start, stop + 1))