#!usr/bin/env python3
"""Question answering module using a pre-trained BERT QA model."""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Find a snippet of text within a reference document to answer
    a given question.

    Args:
        question: string containing the question to answer.
        reference: string containing the reference document to
            search for the answer.

    Returns:
        A string containing the answer, or None if no answer could
        be found within the reference document.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] +
              reference_tokens + ['[SEP]'])

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = (
        [0] * (1 + len(question_tokens) + 1) +
        [1] * (len(reference_tokens) + 1))

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    # skip [CLS] logit at index 0 to avoid selecting the no-answer slot
    short_start = int(tf.argmax(outputs[0][0][1:])) + 1
    short_end = int(tf.argmax(outputs[1][0][1:])) + 1

    if short_end < short_start:
        return None

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer.strip():
        return None

    return answer
