'''
https://github.com/hannamw/gpt2-greater-than/blob/main/utils.py
'''

import torch
from transformers import GPT2TokenizerFast

def get_valid_years(
    tokenizer: GPT2TokenizerFast,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(year_toks) for year_toks in tokens]
    valid = torch.tensor([(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized])
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid]
