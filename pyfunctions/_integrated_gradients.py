# soure: https://github.com/elsanns/xai-nlp-notebooks/blob/master/electra_fine_tune_interpret_captum_ig.ipynb

from captum.attr import (IntegratedGradients,
                         configure_interpretable_embedding_layer,
                         remove_interpretable_embedding_layer)
import torch

def configure_interpretable_embeddings(model):
    """Configure interpretable embedding layer"""
    interpretable_embedding1 = configure_interpretable_embedding_layer(
        model, "bert.embeddings.word_embeddings")
    interpretable_embedding2 = configure_interpretable_embedding_layer(
        model, "bert.embeddings.token_type_embeddings")
    interpretable_embedding3 = configure_interpretable_embedding_layer(
        model,"bert.embeddings.position_embeddings")
    return (interpretable_embedding1,
            interpretable_embedding2,
            interpretable_embedding3)

def get_input_data(interpretable_embedding1, interpretable_embedding2, interpretable_embedding3,
                   text, tokenizer, max_seq_len, device):
    def place_on_device(*tensors):
        tensors_device = []
        for t in tensors:
            tensors_device.append(t.to(device))
        return tuple(tensors_device)

    input_data = place_on_device(*prepare_input(text, tokenizer, max_seq_len))
    input_data += (interpretable_embedding1, interpretable_embedding2, interpretable_embedding3, )
    input_data_embed = prepare_input_embed(*input_data)
    return input_data, input_data_embed

def prepare_input(text, tokenizer, max_seq_len):
    """Prepare ig attribution input: tokenize sample and baseline text."""
    tokenized_text = tokenizer(text, return_tensors="pt",
                               return_attention_mask=True,
                                max_length=max_seq_len,
                                truncation=True, 
                                padding = "max_length", 
                                pad_to_max_length=True,)
    seq_len = tokenized_text["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Construct the baseline (a reference sample).
    # A sequence of [PAD] tokens of length equal to that of the processed sample
    ref_text = tokenizer.pad_token * (seq_len - 2) # special tokens
    tokenized_ref_text = tokenizer(ref_text, return_tensors="pt")
    ref_position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    return (tokenized_text["input_ids"],
            tokenized_text["token_type_ids"],
            position_ids,
            tokenized_ref_text["input_ids"],
            tokenized_ref_text["token_type_ids"],
            ref_position_ids,
            tokenized_text["attention_mask"])

def prepare_input_embed(input_ids, token_type_ids, position_ids,
                        ref_input_ids, ref_token_type_ids, ref_position_ids,
                        attention_mask, interpretable_embedding1, interpretable_embedding2, interpretable_embedding3):
    """Construct input for the modified model"""
    input_ids_embed = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_ids_embed = interpretable_embedding1.indices_to_embeddings(
        ref_input_ids)
    token_type_ids_embed = interpretable_embedding2.indices_to_embeddings(
        token_type_ids)
    ref_token_type_ids_embed = interpretable_embedding2.indices_to_embeddings(
        ref_token_type_ids)
    position_ids_embed = interpretable_embedding3.indices_to_embeddings(
        position_ids)
    ref_position_ids_embed = interpretable_embedding3.indices_to_embeddings(
        ref_position_ids)

    return (input_ids_embed, token_type_ids_embed, position_ids_embed,
            ref_input_ids_embed, ref_token_type_ids_embed,
            ref_position_ids_embed, attention_mask)

def ig_attribute(ig, class_index, input_data_embed):
    return ig.attribute(inputs=input_data_embed[0:3],
                        baselines=input_data_embed[3:6],
                        additional_forward_args=(input_data_embed[6]),
                        target = class_index,
                        return_convergence_delta=True,
                        n_steps=25)

def remove_interpretable_embeddings(model, interpretable_embedding1,
                                    interpretable_embedding2,
                                    interpretable_embedding3):
    """Remove interpretable layer to restore the original model structure"""
    if not \
    type(model.get_input_embeddings()).__name__ == "InterpretableEmbeddingBase":
        return
    remove_interpretable_embedding_layer(model, interpretable_embedding1)
    remove_interpretable_embedding_layer(model, interpretable_embedding2)
    remove_interpretable_embedding_layer(model, interpretable_embedding3)