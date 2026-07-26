#!usr/bin/env python3
"""Interactive command-line question loop.

Prompts the user with `Q:` and echoes `A:` until an exit keyword
(`exit`, `quit`, `goodbye`, `bye`, case insensitive) is entered.
"""
EXIT_WORDS = frozenset(('exit', 'quit', 'goodbye', 'bye'))


if __name__ == '__main__':
    while True:
        question = input('Q: ')

        if question.lower() in EXIT_WORDS:
            print('A: Goodbye')
            break

        print('A:')
