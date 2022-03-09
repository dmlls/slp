# coding: utf-8
"""
Defining global constants
"""

from transformers import AutoTokenizer

# Declare variables
pretrained_model_str = None  # "bert" or "none"
tokenizer = None   # transformers.AutoModelForMaskedLM
vocab = None  # BERT vocabulary
UNK_TOKEN = None
PAD_TOKEN = None
BOS_TOKEN = None
EOS_TOKEN = None
TARGET_PAD = None
DEFAULT_UNK_ID = None

special_tokens = {
    "none": ('<unk>', '<pad>', '<s>', '</s>'),
    "bert": ('[UNK]', '[PAD]', '[CLS]', '[SEP]')
}


def initialize_constants(cfg: dict):
    global pretrained_model_str, tokenizer, vocab, UNK_TOKEN, PAD_TOKEN, \
        BOS_TOKEN, EOS_TOKEN, TARGET_PAD, DEFAULT_UNK_ID
    pretrained_model_str = cfg["model"]["encoder"]["embeddings"]["model"]
    if pretrained_model_str not in ("none", "bert"):
        raise ValueError(f"embeddings from model {pretrained_model_str} not supported")
    UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = special_tokens[pretrained_model_str]
    if pretrained_model_str == "bert":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
        unk_token_id = tokenizer.get_vocab()[UNK_TOKEN]
        # Get vocabulary, sorted by the token ids
        vocab = [token for token in sorted(tokenizer.get_vocab().items(),
                                           key=lambda x: x[1])]
    else:
        tokenizer = None
        unk_token_id = 0
    TARGET_PAD = 0.0
    DEFAULT_UNK_ID = lambda: unk_token_id
