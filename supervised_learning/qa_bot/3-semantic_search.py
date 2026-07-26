#!usr/bin/env python3
"""Semantic search over a corpus of reference documents using the
Universal Sentence Encoder.
"""
import os
import numpy as np
import tensorflow_hub as hub

USE_MODEL_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.

    Args:
        corpus_path: path to the directory containing the reference
            documents (`.md` files) to search.
        sentence: the query sentence to compare against the corpus.

    Returns:
        The full text of the document most semantically similar to
        `sentence`.
    """
    model = hub.load(USE_MODEL_URL)

    filenames = []
    documents = [sentence]

    for filename in sorted(os.listdir(corpus_path)):
        if not filename.endswith('.md'):
            continue

        file_path = os.path.join(corpus_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
        filenames.append(filename)

    embeddings = model(documents).numpy()

    # cosine similarity via normalized inner product; row 0 is the
    # query, so compare it against every document embedding
    query_vec = embeddings[0]
    doc_vecs = embeddings[1:]

    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    similarities = doc_norms @ query_norm
    best_idx = int(np.argmax(similarities))

    best_path = os.path.join(corpus_path, filenames[best_idx])
    with open(best_path, 'r', encoding='utf-8') as f:
        return f.read()
