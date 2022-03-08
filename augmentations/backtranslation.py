"""Back translation script for corpus augmentation."""
import logging
import re
from functools import partial

from num2words import num2words
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)
SOURCE_TEXT = ".data/train.text"
TARGET_AUG_TEXT = ".data/train_aug.text"

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

model_translation = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")


def num_to_words(line):
    """Transform cardinal and ordinal numbers to it's german written equivalent."""
    matches = re.findall("[0-9]+[.]*", line)
    newline = line
    if len(matches):
        for n in matches:
            if n[-1] == ".":
                nword = num2words(n, lang="de", to="ordinal")
            else:
                nword = num2words(n, lang="de", to="cardinal")
            newline = re.sub(n, nword, newline)
    return newline


def translate_de_to_en(line, model, tokenizer):
    """Wrapper function to translate german to english."""
    batch = tokenizer([line], return_tensors="pt")
    gen = model.generate(**batch)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


# create simple function for translation de to en.
de_to_en = partial(translate_de_to_en, model=model_translation, tokenizer=tokenizer)
# get an out of the box en to de translator.
en_to_de = pipeline("translation_en_to_de")

with open(SOURCE_TEXT, 'r') as train_data, open(TARGET_AUG_TEXT, 'a') as aug_train_data:
    for line in tqdm(train_data):
        # Translate to english
        forward_tl = de_to_en(line)
        # Translate back to german
        back_tl = en_to_de(forward_tl)[0]["translation_text"]
        # Clean numeric symbols using helper function.
        new_line = num2words(back_tl)
        # Write to augmented text file so we can train from that later.
        aug_train_data.write(new_line if new_line[-1] == "\n" else new_line + "\n")

logger.info(f"Finished backtranslation dataset augmentation for dataset {SOURCE_TEXT}")
