"""Back translation script for corpus augmentation"""
from functools import partial

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# de_to_en = pipeline("Helsinki-NLP/opus-mt-de-en")


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

model_translation = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")


def translate_de_to_en(line, model, tokenizer):
    batch = tokenizer([line], return_tensors="pt")
    gen = model.generate(**batch)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


de_to_en = partial(translate_de_to_en, model=model_translation, tokenizer=tokenizer)

en_to_de = pipeline("translation_en_to_de")

train_data = open('.data/train.text', 'r')
aug_train_data = open('.data/train_aug.text', 'a')

new_lines = []

for line in tqdm(train_data):
    forward_tl = de_to_en(line)
    # forward_tl = de_to_en(line)["translation_text"]
    back_tl = en_to_de(forward_tl)[0]["translation_text"]
    new_lines.append(back_tl)

with open('.data/train_aug.text', 'a') as aug_train_data:
    for line in new_lines:
        aug_train_data.write(line)

train_data.close()
