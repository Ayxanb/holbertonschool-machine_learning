#!/usr/bin/env python3
"""Module containing the Monte Carlo prediction algorithm for RL."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """ Performs the Monte Carlo algorithm to update value estimates. """
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode = []
        for _ in range(max_steps):
            action = policy(state)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]

            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        states = [step[0] for step in episode]
        for i in range(len(episode) - 1, -1, -1):
            s, r = episode[i]
            G = gamma * G + r
            if s not in states[:i]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
