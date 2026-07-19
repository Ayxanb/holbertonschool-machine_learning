#!/usr/bin/env python3
import transformers
from setup import load_pt2en


class Dataset:
    def __init__(self):
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        pt_tok, en_tok = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt_tok
        self.tokenizer_en = en_tok

    def tokenize_dataset(self, data):
        model_pt = 'neuralmind/bert-base-portuguese-cased'
        model_en = 'bert-base-uncased'

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(model_pt)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(model_en)

        def pt_iter():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iter():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        vocab_size = 2**13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iter(), vocab_size
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iter(), vocab_size
        )

        return tokenizer_pt, tokenizer_en
