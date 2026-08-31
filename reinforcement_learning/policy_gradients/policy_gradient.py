#!/usr/bin/env python3
"""Simple policy function."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy probabilities for a given state and weight matrix.

    Args:
        matrix: State/observation matrix.
        weight: Weight matrix.

    Returns:
        A vector of action probabilities.
    """
    z = np.matmul(matrix, weight)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)

def policy_gradient(state, weight):
    """Compute an action and the gradient of the policy.

    Args:
        state: Matrix representing the current observation.
        weight: Weight matrix.

    Returns:
        A tuple containing:
            - the sampled action
            - the gradient of log(policy(action | state))
    """
    probs = np.asarray(policy(state, weight)).ravel()

    action = np.random.choice(probs.size, p=probs)

    one_hot = np.zeros_like(probs)
    one_hot[action] = 1

    state = np.asarray(state).ravel()
    gradient = np.outer(state, one_hot - probs)

    return action, gradient
