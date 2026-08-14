#!/usr/bin/env python3

"""Module for action selection strategies in reinforcement learning."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select an action using the epsilon-greedy policy.

    Args:
        Q: numpy.ndarray containing the Q-table.
        state: The current state index.
        epsilon: Exploration rate threshold.

    Returns:
        int: The index of the selected action.
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
