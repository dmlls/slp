# coding: utf-8
"""
Defining global constants
"""

# Declare variables
model, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TARGET_PAD, DEFAULT_UNK_ID = [None]*7

special_tokens = {
    "none": ('<unk>', '<pad>', '<s>', '</s>'),
    "bert": ('[UNK]', '[PAD]', '[CLS]', '[SEP]')
}

unk_token_id = {
    "none": 0,
    "bert": 2
}


def initialize_constants(cfg: dict):
    global model, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TARGET_PAD, DEFAULT_UNK_ID
    model = cfg["model"]["encoder"]["embeddings"]["model"]
    if model not in ("none", "bert"):
        raise ValueError(f"embeddings from model {model} not supported")

    UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = special_tokens[model]
    TARGET_PAD = 0.0
    DEFAULT_UNK_ID = lambda: unk_token_id[model]
