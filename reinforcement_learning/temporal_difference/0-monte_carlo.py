#!/usr/bin/env python3
"""Module containing the Monte Carlo prediction algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Performs the Monte Carlo algorithm to update value estimates.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append([state, reward])
            if done:
                break
            state = next_state

        G = 0
        for i in range(len(episode) - 1, -1, -1):
            s, r = episode[i]
            G = gamma * G + r
            visited_states = [step[0] for step in episode[:i]]
            if s not in visited_states:
                V[s] = V[s] + alpha * (G - V[s])

    return V
