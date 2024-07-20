from pyfunctions.patch import *

def patch_context_baseline(rel, irrel, patched_entries, layer_patched_values, sa_module):
    rel = reshape_for_patching(rel, sa_module)
    irrel = reshape_for_patching(irrel, sa_module)
    
    if layer_patched_values is not None:
        layer_patched_values = layer_patched_values[None, :, :, :]

    for entry in patched_entries:
        pos = entry[1]
        att_head = entry[2]

        rel[:, pos, att_head, :] = 0
        irrel[:, pos, att_head, :] = torch.Tensor(layer_patched_values[:, pos, att_head, :]) #substitute with mean response
    
    rel = reshape_post_patching(rel, sa_module)
    irrel = reshape_post_patching(irrel, sa_module)
    return rel, irrel

def prop_attention_patched_baseline(rel, irrel, attention_mask, 
                           head_mask, patched_entries, layer_patched_values, a_module, 
                           att_probs = None,
                           output_att_prob=False,
                           output_context=False):

    
    rel_context, irrel_context, returned_att_probs = prop_self_attention_patched(rel, irrel, 
                                                             attention_mask, 
                                                             head_mask, 
                                                             patched_entries,
                                                             a_module.self, att_probs, output_att_prob)

    output_module = a_module.output
    
    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    
    if output_context:
        # for collecting mean head response
        my_rel_context = reshape_for_patching(rel_tot, a_module.self)
        my_irrel_context = reshape_for_patching(irrel_tot, a_module.self)
        context = {'rel': my_rel_context, 'irrel': my_irrel_context}
        ##
    else:
        context = None
        
    rel_tot, irrel_tot = patch_context_baseline(rel_tot, irrel_tot, patched_entries, layer_patched_values, a_module.self)
    
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)
    
    return rel_out, irrel_out, returned_att_probs, context

def prop_layer_patched_baseline(rel, irrel, attention_mask, head_mask, patched_entries, layer_patched_values, layer_module, att_probs = None, output_att_prob=False, output_context=False):
    
    rel_a, irrel_a, returned_att_probs, context = prop_attention_patched_baseline(rel, irrel, attention_mask, head_mask,
                                                                         patched_entries, layer_patched_values,
                                                                         layer_module.attention, att_probs,
                                                                         output_att_prob, output_context)
    
    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)
    
    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    
    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)
    
    # import pdb; pdb.set_trace()
    
    return rel_out, irrel_out, returned_att_probs, context

def prop_classifier_model_patched_baseline(encoding, model, patched_entries=[], patched_values=None, att_list = None, output_att_prob=False, output_context=False):
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
    rel_context_lst = []
    irrel_context_lst = []
    for i, layer_module in enumerate(encoder_module.layer):
        layer_patched_entries = [p_entry for p_entry in patched_entries if p_entry[0] == i]
        layer_head_mask = head_mask[i]
        att_probs = None
        
        if patched_values is not None:
            layer_patched_values = patched_values[i]
        else:
            layer_patched_values = None
        
        rel_n, irrel_n, returned_att_probs, context = prop_layer_patched_baseline(rel, irrel, extended_attention_mask,
                                                                layer_head_mask, layer_patched_entries,
                                                                layer_patched_values,
                                                                layer_module, att_probs, output_att_prob, output_context)
        normalize_rel_irrel(rel_n, irrel_n)
        rel, irrel = rel_n, irrel_n
        
        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))
        if output_context:
            rel_context_lst.append(context['rel'].cpu().detach().numpy())
            irrel_context_lst.append(context['irrel'].cpu().detach().numpy())
    
    rel_pool, irrel_pool = prop_pooler(rel, irrel, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)
    
    context_dict = {'rel': rel_context_lst, 'irrel': irrel_context_lst}
    
    return rel_out, irrel_out, att_probs_lst, context_dict