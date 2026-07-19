#!/usr/bin/env python3
"""Defines the Dataset class used to load and prep the pt->en
translation corpus for machine translation.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation."""

    def __init__(self):
        """Sets data_train, data_valid, tokenizer_pt and tokenizer_en."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.

        Args:
            data: a Dataset whose examples are formatted as a tuple
                (pt, en):
                    pt is the Tensor containing the Portuguese
                        sentence.
                    en is the Tensor containing the corresponding
                        English sentence.

        Returns:
            tokenizer_pt, tokenizer_en: the trained Portuguese and
                English tokenizers.
        """
        vocab_size = 2 ** 13

        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        base_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        base_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        tokenizer_pt = base_pt.train_new_from_iterator(
            self._batch_iterator(pt_sentences), vocab_size=vocab_size)
        tokenizer_en = base_en.train_new_from_iterator(
            self._batch_iterator(en_sentences), vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en

    @staticmethod
    def _batch_iterator(sentences, batch_size=1000):
        """Yields successive batches of sentences.

        Args:
            sentences: list of str sentences.
            batch_size: number of sentences per yielded batch.

        Yields:
            list of str, a batch of sentences.
        """
        for i in range(0, len(sentences), batch_size):
            yield sentences[i:i + batch_size]
