from pyfunctions.cd import *

def reshape_for_patching(context_layer, sa_module):
    new_shape = context_layer.size()[:-1] + (sa_module.num_attention_heads, sa_module.attention_head_size)
    context_layer = context_layer.view(new_shape)
    return context_layer

def reshape_post_patching(context_layer, sa_module):
    new_shape = context_layer.size()[:-2] + (sa_module.all_head_size,)
    context_layer = context_layer.view(*new_shape)
    return context_layer


def prop_self_attention_patched(rel, irrel, attention_mask, 
                                head_mask, patched_entries,
                                sa_module, att_probs = None, output_att_prob=False):
    
    if att_probs is not None:
        att_probs = att_probs
    else:
        att_probs = get_attention_probs(rel + irrel, attention_mask, head_mask, sa_module)
    
    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)
    
    rel_context = mul_att(att_probs, rel_value, sa_module)
    irrel_context = mul_att(att_probs, irrel_value, sa_module)
    
    if output_att_prob:
        return rel_context, irrel_context, att_probs
    else:
        return rel_context, irrel_context, None
    
def patch_context(rel, irrel, patched_entries, sa_module):
    rel = reshape_for_patching(rel, sa_module)
    irrel = reshape_for_patching(irrel, sa_module)
        
    for entry in patched_entries:
        pos = entry[1]
        att_head = entry[2]

        rel[:, pos, att_head, :] = irrel[:, pos, att_head, :] + rel[:, pos, att_head, :]
        irrel[:, pos, att_head, :] = 0
    
    rel = reshape_post_patching(rel, sa_module)
    irrel = reshape_post_patching(irrel, sa_module)
    return rel, irrel

def patch_context_mean_ablated(rel, irrel, patched_entries, layer_patched_values, sa_module, device):
    rel = reshape_for_patching(rel, sa_module)
    irrel = reshape_for_patching(irrel, sa_module)
    
    if layer_patched_values is not None:
        layer_patched_values = layer_patched_values[None, :, :, :]
        
    for entry in patched_entries:
        pos = entry[1]
        att_head = entry[2]
        
        rel[:, pos, att_head, :] = irrel[:, pos, att_head, :] + rel[:, pos, att_head, :] - torch.Tensor(layer_patched_values[:, pos, att_head, :]).to(device)
        irrel[:, pos, att_head, :] = torch.Tensor(layer_patched_values[:, pos, att_head, :]).to(device)
    
    rel = reshape_post_patching(rel, sa_module)
    irrel = reshape_post_patching(irrel, sa_module)
    
    return rel, irrel

def prop_attention_patched(rel, irrel, attention_mask, 
                           head_mask, patched_entries, layer_patched_values, a_module,
                           device,
                           att_probs=None,
                           output_att_prob=False,
                           output_context=False,
                           mean_ablated=False):
    
    
    rel_context, irrel_context, returned_att_probs = prop_self_attention_patched(rel, irrel, 
                                                             attention_mask, 
                                                             head_mask, 
                                                             patched_entries,
                                                             a_module.self, att_probs, output_att_prob)
    
    
    if output_context:
        # for head output variance analysis
        context = rel_context + irrel_context
        context = reshape_for_patching(context, a_module.self)
        ##
    else:
        context = None
    
    normalize_rel_irrel(rel_context, irrel_context)
    
    output_module = a_module.output
    
    
    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    
    normalize_rel_irrel(rel_dense, irrel_dense)
    
    
    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    
    normalize_rel_irrel(rel_tot, irrel_tot)
    
    # patch the head
    if not mean_ablated:
        rel_tot, irrel_tot = patch_context(rel_tot, irrel_tot, patched_entries, a_module.self)
    else:
        rel_tot, irrel_tot = patch_context_mean_ablated(rel_tot, irrel_tot, patched_entries, layer_patched_values, a_module.self, device)
    
    normalize_rel_irrel(rel_tot, irrel_tot)
    
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)
    
    normalize_rel_irrel(rel_out, irrel_out)
    
    return rel_out, irrel_out, returned_att_probs, context


def prop_layer_patched(rel, irrel, attention_mask, head_mask, patched_entries, layer_patched_values, 
                       layer_module, device, att_probs = None, output_att_prob=False, output_context=False,
                       mean_ablated=False):
    
    # attn module
    rel_a, irrel_a, returned_att_probs, context = prop_attention_patched(rel, irrel, attention_mask, head_mask,
                                                                         patched_entries, layer_patched_values,
                                                                         layer_module.attention, device,
                                                                         att_probs,
                                                                         output_att_prob, output_context,
                                                                         mean_ablated=mean_ablated)
    # linear (dense)
    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    normalize_rel_irrel(rel_id, irrel_id)
    
    # activation
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)
    
    # linear (dense)
    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    normalize_rel_irrel(rel_od, irrel_od)
    
    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    
    normalize_rel_irrel(rel_tot, irrel_tot)
    
    # layer norm
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)
    
    return rel_out, irrel_out, returned_att_probs, context

def prop_encoder_patched(rel, irrel, attention_mask, head_mask, encoder_module, level = 0, att_list = None):
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

def prop_classifier_model_patched(encoding, model, device, patched_entries=[], patched_values=None, 
                                  att_list = None, output_att_prob=False, output_context=False,
                                  mean_ablated=False):
    # patched_entries: attention heads to patch. format: [(level, pos, head)]
    # level: 0-11, pos: 0-511, head: 0-11
    # rel_out: the contribution of the patched_entries
    # irrel_out: the contribution of everything else
    
    embedding_output = get_embeddings(encoding, model.bert)
    input_shape = encoding['input_ids'].size()
    extended_attention_mask = get_extended_attention_mask(attention_mask = encoding['attention_mask'], 
                                                          input_shape = input_shape, 
                                                          bert_model = model.bert,
                                                          device = device)
    
    head_mask = [None] * model.bert.config.num_hidden_layers
    encoder_module = model.bert.encoder
    
    sh = list(embedding_output.shape)
    
    rel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    irrel = torch.zeros(sh, dtype = embedding_output.dtype, device = device)
    
    irrel[:] = embedding_output[:]

    att_probs_lst = []
    context_lst = []
    for i, layer_module in enumerate(encoder_module.layer):
        layer_patched_entries = [p_entry for p_entry in patched_entries if p_entry[0] == i]
        layer_head_mask = head_mask[i]
        att_probs = None
        
        if patched_values is not None:
            layer_patched_values = patched_values[i]
        else:
            layer_patched_values = None
            
        rel_n, irrel_n, returned_att_probs, context = prop_layer_patched(rel, irrel, extended_attention_mask,
                                                                layer_head_mask, layer_patched_entries,
                                                                layer_patched_values,
                                                                layer_module, device, att_probs, output_att_prob,
                                                                output_context,
                                                                mean_ablated=mean_ablated)
        normalize_rel_irrel(rel_n, irrel_n)
        rel, irrel = rel_n, irrel_n
        
        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))
        if output_context:
            context_lst.append(context.cpu().detach().numpy())
    
    rel_pool, irrel_pool = prop_pooler(rel, irrel, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)
    
    return rel_out, irrel_out, att_probs_lst, context_lst

