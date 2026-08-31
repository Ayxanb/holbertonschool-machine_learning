#!/usr/bin/env python3
"""Module that contains the Monte Carlo value estimation algorithm."""

import numpy as np


def monte_carlo(
    env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """Performs the Monte Carlo algorithm for value estimation.

    Args:
        env: Environment instance.
        V (numpy.ndarray): Array of shape (s,) containing value estimates.
        policy (function): Function taking a state and returning action.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: The updated value estimate V.
    """
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode = []
        for _ in range(max_steps):
            action = policy(state)
            res = env.step(action)
            next_state, reward, done = res[0], res[1], res[2]
            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        states = [step[0] for step in episode]
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = gamma * G + reward
            if state not in states[:t]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
