#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm.

    Args:
        env: the environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: a function that takes in a state and returns the next
            action to take.
        episodes: the total number of episodes to train over.
        max_steps: the maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V, the updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)
            next_state, reward, terminated, truncated, _ = step_result
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        states = np.array([step[0] for step in episode], dtype=int)
        rewards = np.array([step[1] for step in episode], dtype=float)
        G = 0.0

        for i in range(len(episode) - 1, -1, -1):
            state = states[i]
            reward = rewards[i]
            G = reward + gamma * G

            if state not in states[:i]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
