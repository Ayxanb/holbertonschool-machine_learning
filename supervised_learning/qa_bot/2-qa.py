#!usr/bin/env python3
"""Answer loop that resolves user questions against a single
reference text using a BERT-based QA model.
"""
question_answer = __import__('0-qa').question_answer

EXIT_WORDS = frozenset(('exit', 'quit', 'goodbye', 'bye'))


def answer_loop(reference):
    """Answer questions from a reference text.

    Args:
        reference: string containing the reference text to search
            for answers.

    Returns:
        None. Runs an interactive loop until an exit keyword is
        entered.
    """
    while True:
        question = input('Q: ')

        if question.lower() in EXIT_WORDS:
            print('A: Goodbye')
            break

        answer = question_answer(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
