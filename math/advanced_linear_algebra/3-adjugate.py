#!/usr/bin/env python3

'''
This module contains `adjugate` function
'''


def adjugate(matrix):
    '''
    Returns the adjugate matrix
    '''
    cofactor = __import__('2-cofactor').cofactor

    cofactor_matrix = cofactor(matrix)
    return [list(i) for i in zip(*matrix)]
