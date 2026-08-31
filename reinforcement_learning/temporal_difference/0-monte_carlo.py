#!/usr/bin/env python3
"""Monte Carlo algorithm module"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for estimating a value function.

    Args:
        env: the environment instance.
        V (numpy.ndarray): array of shape (s,) containing the value
            estimate for each state.
        policy (callable): a function that takes in a state and returns
            the next action to take.
        episodes (int): the total number of episodes to train over.
        max_steps (int): the maximum number of steps per episode.
        alpha (float): the learning rate.
        gamma (float): the discount rate.

    Returns:
        numpy.ndarray: V, the updated value estimate.
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append([state, reward])
            state = next_state
            if done or truncated:
                break

        episode = np.array(episode, dtype=int)
        G = 0

        for i, (state, reward) in enumerate(episode[::-1]):
            G = gamma * G + reward
            visited_earlier = state in episode[:episode.shape[0] - 1 - i, 0]
            if not visited_earlier:
                V[state] = V[state] + alpha * (G - V[state])

    return V
