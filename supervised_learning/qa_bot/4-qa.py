#!usr/bin/env python3
"""Answer loop that resolves user questions against a corpus of
multiple reference documents, selecting the most relevant document
via semantic search before running BERT QA over it.
"""
semantic_search = __import__('3-semantic_search').semantic_search
bert_qa = __import__('0-qa').question_answer

EXIT_WORDS = frozenset(('exit', 'quit', 'goodbye', 'bye'))


def question_answer(corpus_path):
    """Answer questions from multiple reference texts.

    Args:
        corpus_path: path to the corpus of reference documents.

    Returns:
        None. Runs an interactive loop until an exit keyword is
        entered.
    """
    while True:
        question = input('Q: ')

        if question.lower() in EXIT_WORDS:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)
        answer = bert_qa(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
