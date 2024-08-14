from pyfunctions.patch import *
from transformers.activations import NewGELUActivation
from fancy_einsum import einsum
import pdb

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

# TODO: ensure that these conventions match the ones used by BERT on some level. We may be transposed, since TransformerLens multiplies on the right.
class GPTAttentionWrapper():
    def __init__(self, attn_module):
        self.attn_module = attn_module

    # TODO: einsum notation, once we are more sure about what the dimensions are
    def query(self, embedding):
        pdb.set_trace()
        return einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", embedding, self.attn_module.W_Q) + self.attn_module.b_Q

    def key(self, embedding):
        return einsum("batch query_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", embedding, self.attn_module.W_K) + self.attn_module.b_K

def patch_context_hh(rel, irrel, source_node_list, target_nodes, level, sa_module, device):
    rel = reshape_for_patching(rel, sa_module)
    irrel = reshape_for_patching(irrel, sa_module)
    
    target_nodes_at_level = [node for node in target_nodes if node[0] == level]
    target_decomps = []
    
    for s_ind, sn_list in enumerate(source_node_list):
        out_shape = (len(target_nodes_at_level), sa_module.attention_head_size)
        
        rel_st = torch.zeros(out_shape, dtype = rel.dtype, device = device)
        irrel_st = torch.zeros(out_shape, dtype = rel.dtype, device = device)
        
        for t_ind, t in enumerate(target_nodes_at_level):
            if t[0] == level:
                t_pos = t[1]
                t_head = t[2]

                rel_st[t_ind, :] = rel[s_ind, t_pos, t_head, :]
                irrel_st[t_ind, :] = irrel[s_ind, t_pos, t_head, :]
        
        target_decomps.append((rel_st.detach().cpu().numpy(), irrel_st.detach().cpu().numpy()))
        
        for entry in sn_list:
            if entry[0] == level:
                pos = entry[1]
                att_head = entry[2]

                rel[s_ind, pos, att_head, :] = rel[s_ind, pos, att_head, :] + irrel[s_ind, pos, att_head, :]
                irrel[s_ind, pos, att_head, :] = 0

    
    rel = reshape_post_patching(rel, sa_module)
    irrel = reshape_post_patching(irrel, sa_module)
    
    return rel, irrel, target_decomps

def patch_context_hh_mean_ablated(rel, irrel, source_node_list, target_nodes, level, layer_patched_values, sa_module, device):
    rel = reshape_for_patching(rel, sa_module)
    irrel = reshape_for_patching(irrel, sa_module)
    
    target_nodes_at_level = [node for node in target_nodes if node[0] == level]
    target_decomps = []
    
    if layer_patched_values is not None:
        layer_patched_values = layer_patched_values[None, :, :, :]

    for s_ind, sn_list in enumerate(source_node_list):
        out_shape = (len(target_nodes_at_level), sa_module.attention_head_size)
        
        rel_st = torch.zeros(out_shape, dtype = rel.dtype, device = device)
        irrel_st = torch.zeros(out_shape, dtype = rel.dtype, device = device)

        for t_ind, t in enumerate(target_nodes_at_level):
            if t[0] == level:
                t_pos = t[1]
                t_head = t[2]
                rel_st[t_ind, :] = rel[s_ind, t_pos, t_head, :]
                irrel_st[t_ind, :] = irrel[s_ind, t_pos, t_head, :]

        
        target_decomps.append((rel_st.detach().cpu().numpy(), irrel_st.detach().cpu().numpy()))
        
        for entry in sn_list:
            if entry[0] == level:
                pos = entry[1]
                att_head = entry[2]
                
                rel[s_ind, pos, att_head, :] = irrel[s_ind, pos, att_head, :] + rel[s_ind, pos, att_head, :] - torch.Tensor(layer_patched_values[:, pos, att_head, :]).to(device)
                irrel[s_ind, pos, att_head, :] = torch.Tensor(layer_patched_values[:, pos, att_head, :]).to(device)

    
    rel = reshape_post_patching(rel, sa_module)
    irrel = reshape_post_patching(irrel, sa_module)
    
    return rel, irrel, target_decomps

def prop_self_attention_hh(rel, irrel, attention_mask, 
                           head_mask, source_node_list, target_nodes, 
                           level, sa_module, device, att_probs = None, output_att_prob=False):
    
    if att_probs is not None:
        att_probs = att_probs
    else:
        att_probs = get_attention_probs(rel[0].unsqueeze(0) + irrel[0].unsqueeze(0), attention_mask, head_mask, sa_module)

    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)

    rel_context = mul_att(att_probs, rel_value, sa_module)

    irrel_context = mul_att(att_probs, irrel_value, sa_module)
        
    if output_att_prob:
        return rel_context, irrel_context, att_probs
    else:
        return rel_context, irrel_context, None
    
def prop_attention_hh(rel, irrel, attention_mask, 
                      head_mask, source_node_list, target_nodes, level,
                      layer_patched_values,
                      a_module, device, att_probs = None, output_att_prob=False, mean_ablated=False):
    
    rel_context, irrel_context, returned_att_probs = prop_self_attention_hh(rel, irrel, 
                                                                        attention_mask, 
                                                                        head_mask, 
                                                                        source_node_list,
                                                                        target_nodes,
                                                                        level,
                                                                        a_module.self,
                                                                        device,
                                                                        att_probs,
                                                                        output_att_prob=output_att_prob)
    normalize_rel_irrel(rel_context, irrel_context)
    
    output_module = a_module.output
    
    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    
    normalize_rel_irrel(rel_dense, irrel_dense)
    
    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    
    normalize_rel_irrel(rel_tot, irrel_tot)
    
    if not mean_ablated:
        rel_tot, irrel_tot, target_decomps = patch_context_hh(rel_tot, irrel_tot, source_node_list,
                                                              target_nodes, level, a_module.self, device)
    else:
        rel_tot, irrel_tot, target_decomps = patch_context_hh_mean_ablated(rel_tot, irrel_tot, source_node_list,
                                                                           target_nodes, level, layer_patched_values,
                                                                           a_module.self, device)
    
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)

    normalize_rel_irrel(rel_out, irrel_out)
    
    return rel_out, irrel_out, target_decomps, returned_att_probs

def prop_BERT_layer_hh(rel, irrel, attention_mask, head_mask, 
                  source_node_list, target_nodes, level, layer_patched_values,
                  layer_module, device, att_probs = None, output_att_prob=False, mean_ablated=False):
    
    rel_a, irrel_a, target_decomps, returned_att_probs = prop_attention_hh(rel, irrel, attention_mask, 
                                                                           head_mask, source_node_list, 
                                                                           target_nodes, level, layer_patched_values,
                                                                           layer_module.attention,
                                                                           device,
                                                                           att_probs, output_att_prob, mean_ablated=mean_ablated)

    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    normalize_rel_irrel(rel_id, irrel_id)
    
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)
    
    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    normalize_rel_irrel(rel_od, irrel_od)
    
    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    normalize_rel_irrel(rel_tot, irrel_tot)

    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)
    
    
    return rel_out, irrel_out, target_decomps, returned_att_probs

def prop_GPT_layer_hh(rel, irrel, attention_mask, head_mask, 
                  source_node_list, target_nodes, level, layer_patched_values,
                  layer_module, device, att_probs = None, output_att_prob=False, mean_ablated=False):
    
    # TODO: there should be some kind of casework for the folded layernorm,
    # if we want to perfectly apples-to-apples reproduce the IOI paper. 
    rel_ln, irrel_ln = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(layer_module.ln1))
    
    # what the BERT model calls attention is what this model calls attention, plus a linear layer,
    # since it's an encoder model. Since this is a decoder model, we only need the "self attention" function,
    # and also we have to do the second layer norm ourselves out here.
    rel_attn_residual, irrel_attn_residual, returned_att_probs = prop_self_attention_hh(rel_ln, irrel_ln, attention_mask, 
                                                                           head_mask, source_node_list, 
                                                                           target_nodes, level,
                                                                           GPTAttentionWrapper(layer_module.attn),
                                                                           device,
                                                                           att_probs, output_att_prob)
    normalize_rel_irrel(rel_attn_residual, irrel_attn_residual)
    rel_attn, irrel_attn = rel + rel_attn_residual, irrel + irrel_attn_residual

    normalize_rel_irrel(rel_attn, irrel_attn)
    #patch_context_hh function would go here # TODO figure out what that does


    rel_mid, irrel_mid = prop_layer_norm(rel_attn, irrel_attn, GPTLayerNormWrapper(layer_module.ln2))

    # MLP

    rel_after_w_in, irrel_after_w_in = prop_linear_core(rel_mid, irrel_mid, layer_module.mlp.W_in, layer_module.mlp.b_in)
    normalize_rel_irrel(rel_after_w_in, irrel_after_w_in)
    
    # since GELU activation is stateless, it's not an attribute of the layer module
    rel_act, irrel_act = prop_act(rel_after_w_in, irrel_after_w_in, NewGELUActivation())     
    rel_mlp_residual, irrel_mlp_residual = prop_linear_core(rel_act, irrel_act, layer_module.mlp.W_out, layer_module.mlp.b_out)
    normalize_rel_irrel(rel_mlp_residual, irrel_mlp_residual)
    rel_out, irrel_out = rel_mid + rel_mlp_residual, irrel_mid + irrel_mlp_residual
    normalize_rel_irrel(rel_out, irrel_out)

    target_decomps = None #TODO, has to do with the output of the patch_context_hh function.

    # there is not a layernorm at the end of this block, unlike in BERT    

    return rel_out, irrel_out, target_decomps, returned_att_probs

def prop_BERT_hh(encoding, model, source_node_list, target_nodes, device,
                             patched_values=None, att_list = None, output_att_prob=False, mean_ablated=False):
    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding['input_ids'].size()
    extended_attention_mask = get_extended_attention_mask(attention_mask = encoding['attention_mask'], 
                                                          input_shape = input_shape, 
                                                          model = model,
                                                          device = device)
    
    head_mask = [None] * model.bert.config.num_hidden_layers
    encoder_module = model.bert.encoder
    
    sh = list(embedding_output.shape)
    sh[0] = len(source_node_list)
    
    rel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    irrel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    
    irrel[:] = embedding_output[:]
    
    target_decomps = []
    att_probs_lst = []
    for i, layer_module in enumerate(encoder_module.layer):
        layer_head_mask = head_mask[i]
        att_probs = None
        
        if patched_values is not None:
            layer_patched_values = patched_values[i] #[512, 12, 64]
        else:
            layer_patched_values = None
            
        rel_n, irrel_n, layer_target_decomps, returned_att_probs = prop_BERT_layer_hh(rel, irrel, extended_attention_mask, 
                                                                                 layer_head_mask, source_node_list, 
                                                                                 target_nodes, i, 
                                                                                 layer_patched_values,
                                                                                 layer_module, 
                                                                                 device,
                                                                                 att_probs, output_att_prob,
                                                                                 mean_ablated=mean_ablated)
        target_decomps.append(layer_target_decomps)
        normalize_rel_irrel(rel_n, irrel_n)
        rel, irrel = rel_n, irrel_n
        
        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))
    
    rel_pool, irrel_pool = prop_pooler(rel, irrel, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)
    
    out_decomps = []

    for i, sn_list in enumerate(source_node_list):
        rel_vec = rel_out[i, :].detach().cpu().numpy()
        irrel_vec = irrel_out[i, :].detach().cpu().numpy()
        
        out_decomps.append((rel_vec, irrel_vec))
    
    return out_decomps, target_decomps, att_probs_lst


def prop_GPT_hh(encoding, model, source_node_list, target_nodes, device,
                             patched_values=None, att_list = None, output_att_prob=False, mean_ablated=False):
    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding['input_ids'].size()

    extended_attention_mask = get_extended_attention_mask(encoding['attention_mask'], 
                                                          input_shape, 
                                                          model,
                                                          device)
    
    head_mask = [None] * len(model.blocks)
    
    sh = list(embedding_output.shape)
    sh[0] = len(source_node_list)
    
    rel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    irrel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    
    irrel[:] = embedding_output[:]
    
    target_decomps = []
    att_probs_lst = []
    for i, layer_module in enumerate(model.blocks):
        layer_head_mask = head_mask[i]
        att_probs = None
        
        if patched_values is not None:
            layer_patched_values = patched_values[i] #[512, 12, 64]
        else:
            layer_patched_values = None
            
        rel, irrel, layer_target_decomps, returned_att_probs = prop_GPT_layer_hh(rel, irrel, extended_attention_mask, 
                                                                                 layer_head_mask, source_node_list, 
                                                                                 target_nodes, i, 
                                                                                 layer_patched_values,
                                                                                 layer_module, 
                                                                                 device,
                                                                                 att_probs, output_att_prob,
                                                                                 mean_ablated=mean_ablated)
        target_decomps.append(layer_target_decomps)
        # normalize_rel_irrel(rel_n, irrel_n)
        # rel, irrel = rel_n, irrel_n
        
        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))
    rel, irrel = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(model.ln_final))
    rel_out, irrel_out = prop_GPT_unembed(rel, irrel, model.unembed)
    
    out_decomps = []

    for i, sn_list in enumerate(source_node_list):
        rel_vec = rel_out[i, :].detach().cpu().numpy()
        irrel_vec = irrel_out[i, :].detach().cpu().numpy()
        
        out_decomps.append((rel_vec, irrel_vec))
    
    return out_decomps, target_decomps, att_probs_lst

import transformers # hack, just so the isinstance works
import transformer_lens
def prop_model_hh_batched(encoding, model, source_node_list, target_nodes, device,
                                     patched_values=None, 
                                     num_at_time = 64, n_layers = 12, att_list = None, output_att_prob=False,
                                     mean_ablated=False):
    
    out_decomps = []
    target_decomps = [[] for i in range(n_layers)]
    
    n_source_lists = len(source_node_list)
    n_batches = int((n_source_lists + (num_at_time - 1)) / num_at_time)

    for b_no in range(n_batches):
        b_st = b_no * num_at_time
        b_end = min(b_st + num_at_time, n_source_lists)
        if isinstance(model, transformers.models.bert.modeling_bert.BertForSequenceClassification):
            layer_out_decomps, layer_target_decomps, att_probs_lst = prop_BERT_hh(encoding, model, 
                                                                            source_node_list[b_st: b_end],
                                                                            target_nodes, device,
                                                                            patched_values,
                                                                            att_list=att_list,
                                                                            output_att_prob=output_att_prob,
                                                                            mean_ablated=mean_ablated)
        elif isinstance(model, transformer_lens.HookedTransformer):
            layer_out_decomps, layer_target_decomps, att_probs_lst = prop_GPT_hh(encoding, model, 
                                                                            source_node_list[b_st: b_end],
                                                                            target_nodes, device,
                                                                            patched_values,
                                                                            att_list=att_list,
                                                                            output_att_prob=output_att_prob,
                                                                            mean_ablated=mean_ablated)

        out_decomps = out_decomps + layer_out_decomps
        target_decomps = [target_decomps[i] + layer_target_decomps[i] for i in range(n_layers)]
    
    return out_decomps, target_decomps