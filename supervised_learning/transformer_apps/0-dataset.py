#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


def load_pt2en(split):
    """Loads the portuguese to english dataset from TFDS."""
    return tfds.load('ted_hrlr_translate/pt_to_en',
                     split=split,
                     as_supervised=True)


class Dataset:
    """Dataset class for preparing machine translation data."""

    def __init__(self):
        """Initializes the Dataset instance."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        tok_pt, tok_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tok_pt
        self.tokenizer_en = tok_en

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en) Tensors.

        Returns:
            tokenizer_pt: The trained Portuguese tokenizer.
            tokenizer_en: The trained English tokenizer.
        """
        model_pt = 'neuralmind/bert-base-portuguese-cased'
        model_en = 'bert-base-uncased'

        # Load pre-trained tokenizers
        tokenizer_pt = AutoTokenizer.from_pretrained(model_pt)
        tokenizer_en = AutoTokenizer.from_pretrained(model_en)

        # Create generators to yield decoded string sentences
        def pt_iter():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iter():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Train new tokenizers based on the pre-trained ones
        vocab_size = 2**13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iter(), vocab_size
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iter(), vocab_size
        )

        return tokenizer_pt, tokenizer_en
