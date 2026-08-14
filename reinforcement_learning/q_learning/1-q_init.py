#!/usr/bin/env python3

"""Module for Q-learning initialization and table management."""

import numpy as np


def q_init(env):
    """Initialize the Q-table with zeros for the given environment.

    Args:
        env: The Gymnasium environment instance.

    Returns:
        numpy.ndarray: A 2D array of zeros with shape
            (number_of_states, number_of_actions).
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return np.zeros((num_states, num_actions))
