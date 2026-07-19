#!/usr/bin/env python3
"""Defines the create_masks function used to build the padding and
look-ahead masks required for transformer training/validation.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) containing
            the input sentence.
        target: tf.Tensor of shape (batch_size, seq_len_out)
            containing the target sentence.

    Returns:
        encoder_mask, combined_mask, decoder_mask:
            encoder_mask: tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in), used in the encoder.
            combined_mask: tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out), the maximum
                between the look-ahead mask and the decoder target
                padding mask, used in the decoder's 1st attention
                block.
            decoder_mask: tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in), used in the decoder's
                2nd attention block.
    """
    seq_len_out = tf.shape(target)[1]

    encoder_mask = _padding_mask(inputs)
    decoder_mask = _padding_mask(inputs)

    look_ahead_mask = _look_ahead_mask(seq_len_out)
    target_padding_mask = _padding_mask(target)
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask


def _padding_mask(seq):
    """Builds a padding mask that marks the zero-padded positions of
    seq with a 1.

    Args:
        seq: tf.Tensor of shape (batch_size, seq_len) of token ids,
            where 0 indicates a padding token.

    Returns:
        tf.Tensor of shape (batch_size, 1, 1, seq_len).
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def _look_ahead_mask(size):
    """Builds a look-ahead mask that masks future positions in a
    sequence of length size.

    Args:
        size: int, the length of the target sequence.

    Returns:
        tf.Tensor of shape (size, size).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask
