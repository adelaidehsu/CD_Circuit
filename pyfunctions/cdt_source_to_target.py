from pyfunctions.cdt_from_source_nodes import *
from transformers.activations import NewGELUActivation
import pdb
from pyfunctions.wrappers import GPTAttentionWrapper, GPTLayerNormWrapper, OutputDecomposition

def calculate_contributions(rel, irrel, source_node_list, target_nodes, level, sa_module, device):
    rel = reshape_separate_attention_heads(rel, sa_module)
    irrel = reshape_separate_attention_heads(irrel, sa_module)
    target_nodes_at_level = [node for node in target_nodes if node[0] == level]
    target_decomps = []

    #for s_ind, _ in enumerate(source_node_list):
    out_shape = (len(target_nodes_at_level), sa_module.attention_head_size)

    rel_st = torch.zeros(out_shape, dtype = rel.dtype, device = device)
    irrel_st = torch.zeros(out_shape, dtype = rel.dtype, device = device)

    for t_ind, t in enumerate(target_nodes_at_level):
        t_pos = t[1]
        t_head = t[2]
        rel_st[t_ind, :] = rel[:, t_pos, t_head, :]
        irrel_st[t_ind, :] = irrel[:, t_pos, t_head, :]

    target_decomps.append((rel_st.detach().cpu().numpy(), irrel_st.detach().cpu().numpy()))
    
    return target_decomps

# This function handles what is usually called the attention mechanism, up to the point
# where the softmax'd attention pattern is multiplied by the value vectors.
# It does not include the output matrix.
# The reason for this arrangement is due to the way that HF BERT organizes its internal modules.
# BERT's attention mechanism is split into modules 'self' and 'output', with this corresponding to the former.
# Additionally, BERT and GPT differ in the placement of their LayerNorms, with BERT doing one before addition to the residual,
# and GPT doing one after, so it's messy to write a single "attention" function which handles both.
# However, the contents of this function are common to both BERT and GPT.
def prop_attention_no_output_hh(rel, irrel, attention_mask, 
                           head_mask, sa_module, att_probs = None,
                           output_att_prob=False):
    if att_probs is not None:
        att_probs = att_probs
    else:
        att_probs = get_attention_probs(rel[0].unsqueeze(0) + irrel[0].unsqueeze(0), attention_mask, head_mask, sa_module)

    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)

    rel_context = mul_att(att_probs, rel_value, sa_module)

    irrel_context = mul_att(att_probs, irrel_value, sa_module)
    
    #print(sa_module(rel+irrel)[0].all() == (rel_context+irrel_context).all()) #passed

    if output_att_prob:
        return rel_context, irrel_context, att_probs
    else:
        return rel_context, irrel_context, None
    
def prop_BERT_attention_hh(rel, irrel, attention_mask, 
                      head_mask, source_node_list, target_nodes, level,
                      layer_mean_acts,
                      a_module, device, att_probs = None, output_att_prob=False, set_irrel_to_mean=False):
    
    rel_context, irrel_context, returned_att_probs = prop_attention_no_output_hh(rel, irrel, 
                                                                        attention_mask, 
                                                                        head_mask, 
                                                                        a_module.self,
                                                                        att_probs,
                                                                        output_att_prob=output_att_prob)
    
    #print(a_module.self(rel+irrel)[0].all() == (rel_context+irrel_context).all()) #passed
    
    #tmp = rel_context + irrel_context
    normalize_rel_irrel(rel_context, irrel_context)
    #print((rel_context+irrel_context).all() == tmp.all()) #passed
    
    output_module = a_module.output
    
    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    #print(output_module.dense(rel_context+irrel_context).all() == (rel_dense+irrel_dense).all()) #passed
    
    #tmp = rel_dense + irrel_dense
    normalize_rel_irrel(rel_dense, irrel_dense)
    #print((rel_dense+irrel_dense).all() == tmp.all()) #passed
    
    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    
    #tmp = rel_tot + irrel_tot
    normalize_rel_irrel(rel_tot, irrel_tot)
    #print((rel_tot+irrel_tot).all() == tmp.all()) #passed

    # now that we've calculated the output of the attention mechanism, set desired inputs to "relevant"
    #tmp = rel_tot + irrel_tot
    rel_tot, irrel_tot = set_rel_at_source_nodes(rel_tot, irrel_tot, source_node_list, layer_mean_acts, a_module.self, set_irrel_to_mean, device)
    #print((rel_tot+irrel_tot).all() == tmp.all()) #passed
    target_decomps = calculate_contributions(rel_tot, irrel_tot, source_node_list,
                                                                           target_nodes, level,
                                                                           a_module.self, device=device)
    #print((rel_tot+irrel_tot).all() == tmp.all()) #passed
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)
    #print(output_module.LayerNorm(rel_tot+irrel_tot).all() == (rel_out+irrel_out).all()) #passed
    
    #tmp = rel_out + irrel_out
    normalize_rel_irrel(rel_out, irrel_out)
    #print((rel_out+irrel_out).all() == tmp.all()) #passed
    
    return rel_out, irrel_out, target_decomps, returned_att_probs

def prop_BERT_layer_hh(rel, irrel, attention_mask, head_mask, 
                  source_node_list, target_nodes, level, layer_mean_acts,
                  layer_module, device, att_probs = None, output_att_prob=False, set_irrel_to_mean=False):
    
    rel_a, irrel_a, target_decomps, returned_att_probs = prop_BERT_attention_hh(rel, irrel, attention_mask, 
                                                                           head_mask, source_node_list, 
                                                                           target_nodes, level, layer_mean_acts,
                                                                           layer_module.attention,
                                                                           device,
                                                                           att_probs, output_att_prob, set_irrel_to_mean=set_irrel_to_mean)

    #print(layer_module.attention(rel+irrel)[0].all() == (rel_a+irrel_a).all()) #passed
    
    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    #print(i_module.dense(rel_a+irrel_a).all() == (rel_id+irrel_id).all()) #passed
    normalize_rel_irrel(rel_id, irrel_id)
    
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)
    #print(i_module.intermediate_act_fn(rel_id+irrel_id).all() == (rel_iact+irrel_iact).all()) #passed

    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    #print(o_module.dense(rel_iact+irrel_iact).all() == (rel_od+irrel_od).all()) #passed
    normalize_rel_irrel(rel_od, irrel_od)
    
    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    normalize_rel_irrel(rel_tot, irrel_tot)

    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)
    #print(o_module.LayerNorm(rel_tot+irrel_tot).all() == (rel_out+irrel_out).all()) #passed
    
    return rel_out, irrel_out, target_decomps, returned_att_probs

def prop_GPT_layer_hh(rel, irrel, attention_mask, head_mask, 
                  source_node_list, target_nodes, level, layer_mean_acts,
                  layer_module, device, att_probs = None, output_att_prob=False, set_irrel_to_mean=False):
    # TODO: there should be some kind of casework for the folded layernorm,
    # if we want to perfectly apples-to-apples reproduce the IOI paper. 
    rel_ln, irrel_ln = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(layer_module.ln1))
    attn_wrapper = GPTAttentionWrapper(layer_module.attn)
    rel_summed_values, irrel_summed_values, returned_att_probs = prop_attention_no_output_hh(rel_ln, irrel_ln, attention_mask, 
                                                                           head_mask,
                                                                           attn_wrapper,

                                                                           att_probs, output_att_prob)

    rel_attn_residual, irrel_attn_residual = prop_linear(rel_summed_values, irrel_summed_values, attn_wrapper.output)
    # now that we've calculated the output of the attention mechanism, set desired inputs to "relevant"
    rel_attn_residual, irrel_attn_residual = set_rel_at_source_nodes(rel_attn_residual, irrel_attn_residual, source_node_list, layer_mean_acts, attn_wrapper, set_irrel_to_mean, device)
    layer_target_decomps = calculate_contributions(rel_attn_residual, irrel_attn_residual, source_node_list,
                                                                           target_nodes, level,
                                                                           attn_wrapper, device=device)
    rel_mid, irrel_mid = rel + rel_attn_residual, irrel + irrel_attn_residual
    rel_mid_norm, irrel_mid_norm = prop_layer_norm(rel_mid, irrel_mid, GPTLayerNormWrapper(layer_module.ln2))
    

    # MLP

    rel_after_w_in, irrel_after_w_in = prop_linear_core(rel_mid_norm, irrel_mid_norm, layer_module.mlp.W_in, layer_module.mlp.b_in)
    normalize_rel_irrel(rel_after_w_in, irrel_after_w_in)
    
    # since GELU activation is stateless, it's not an attribute of the layer module
    rel_act, irrel_act = prop_act(rel_after_w_in, irrel_after_w_in, NewGELUActivation())     
    rel_mlp_residual, irrel_mlp_residual = prop_linear_core(rel_act, irrel_act, layer_module.mlp.W_out, layer_module.mlp.b_out)
    normalize_rel_irrel(rel_mlp_residual, irrel_mlp_residual)
    rel_out, irrel_out = rel_mid + rel_mlp_residual, irrel_mid + irrel_mlp_residual
    normalize_rel_irrel(rel_out, irrel_out)

    # there is not a layernorm at the end of this block, unlike in BERT    

    return rel_out, irrel_out, layer_target_decomps, returned_att_probs

def prop_BERT_hh(encoding, model, source_node_list, target_nodes, device,
                             mean_acts=None, output_att_prob=False, set_irrel_to_mean=False):
    
    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding['input_ids'].size()
    extended_attention_mask = get_extended_attention_mask(encoding.attention_mask, 
                                                            input_shape, 
                                                            model,
                                                            device)
    
    head_mask = [None] * model.bert.config.num_hidden_layers
    encoder_module = model.bert.encoder
    
    sh = list(embedding_output.shape)
    #sh[0] = len(source_node_list)

    target_decomps = []
    att_probs_lst = []
    out_decomps = []
    
    # source_node_list should be a List[List], sotring batches of source nodes to ablate
    for source_node_batch in source_node_list:
        
        rel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
        irrel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
        irrel[:] = embedding_output[:]
        
        batch_target_decomps = []
        
        for i, layer_module in enumerate(encoder_module.layer):
            # select source nodes on layer i
            layer_source_node_list = [x for x in source_node_batch if x[0] == i]

            layer_head_mask = head_mask[i]
            att_probs = None

            if mean_acts is not None:
                layer_mean_acts = mean_acts[i] #[512, 12, 64]
            else:
                layer_mean_acts = None

            rel_n, irrel_n, layer_target_decomps, returned_att_probs = prop_BERT_layer_hh(rel, irrel, extended_attention_mask, 
                                                                                     layer_head_mask, layer_source_node_list, # only ablate source node on this layer
                                                                                     target_nodes, i, 
                                                                                     layer_mean_acts,
                                                                                     layer_module, 
                                                                                     device,
                                                                                     att_probs, output_att_prob,
                                                                                     set_irrel_to_mean=set_irrel_to_mean)
            #print(layer_module(rel+irrel)[0].all() == (rel_n+irrel_n).all()) #passed
            #target_decomps.append(layer_target_decomps)
            batch_target_decomps.extend(layer_target_decomps)
            
            normalize_rel_irrel(rel_n, irrel_n)
            rel, irrel = rel_n, irrel_n

            if output_att_prob:
                att_probs_lst.append(returned_att_probs.squeeze(0))

        rel_pool, irrel_pool = prop_pooler(rel, irrel, model.bert.pooler)
        rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)
    
    

        rel_vec = rel_out.detach().cpu().numpy()
        irrel_vec = irrel_out.detach().cpu().numpy()
        
        out_decomps.append((rel_vec, irrel_vec))
        target_decomps.append(batch_target_decomps)
    
    return out_decomps, target_decomps, att_probs_lst


# Single function, analogous to prop_BERT_hh, which should perform the tasks
#  of what is currently called prop_BERT_hh, prop_classifier_model_patched, and prop_classifier_model.
# In order to get head-to-head contribution, pass in source and target nodes, and look at return val target_decomps.
# In order to get source to logits contribution, pass in source nodes, and look at return val out_decomps.
def prop_GPT(encoding_idxs, extended_attention_mask, model, source_node_list, target_nodes, device,
                             mean_acts=None, att_list = None, output_att_prob=False, set_irrel_to_mean=False):
    
    embedding_output = model.embed(encoding_idxs) + model.pos_embed(encoding_idxs) 
    
    
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
        
        if mean_acts is not None:
            layer_mean_acts = mean_acts[i] #[512, 12, 64]
        else:
            layer_mean_acts = None
            
        rel, irrel, layer_target_decomps, returned_att_probs = prop_GPT_layer_hh(rel, irrel, extended_attention_mask, 
                                                                                 layer_head_mask, source_node_list, 
                                                                                 target_nodes, i, 
                                                                                 layer_mean_acts,
                                                                                 layer_module, 
                                                                                 device,
                                                                                 att_probs, output_att_prob,
                                                                                 set_irrel_to_mean=set_irrel_to_mean)
        target_decomps.append(layer_target_decomps)
        
        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))
    rel, irrel = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(model.ln_final))
    rel_out, irrel_out = prop_GPT_unembed(rel, irrel, model.unembed)
    
    out_decomps = []

    for i, sn_list in enumerate(source_node_list):
        rel_vec = rel_out[i, :].detach().cpu().numpy()
        irrel_vec = irrel_out[i, :].detach().cpu().numpy()
        
        out_decomps.append(OutputDecomposition(rel_vec, irrel_vec))
    
    return out_decomps, target_decomps, att_probs_lst

'''
This is different from running a model on a batch of input data.
Instead it calculates the decomposition relative to many source nodes at the same time.
'''

def batch_run(prop_model_fn, source_node_list, num_at_time=64, n_layers=12):
    
    out_decomps = []
    target_decomps = [[] for i in range(n_layers)]
    
    n_source_lists = len(source_node_list)
    n_batches = int((n_source_lists + (num_at_time - 1)) / num_at_time)

    for b_no in range(n_batches):
        b_st = b_no * num_at_time
        b_end = min(b_st + num_at_time, n_source_lists)
        batch_out_decomps, batch_target_decomps, _ = prop_model_fn(source_node_list[b_st: b_end])

        out_decomps = out_decomps + batch_out_decomps
        target_decomps = [target_decomps[i] + batch_target_decomps[i] for i in range(n_layers)]
    
    return out_decomps, target_decomps