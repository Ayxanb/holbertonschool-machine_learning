#!/usr/bin/env python3
"""Monte Carlo value estimation."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform Monte Carlo prediction to update a value estimate.

    Args:
        env: Environment instance.
        V: NumPy array containing the value estimate for each state.
        policy: Function taking a state and returning an action.
        episodes: Number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        The updated value estimate V.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        trajectory = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        trajectory = np.array(trajectory, dtype=int)
        G = 0
        for state, reward in reversed(trajectory):
            G = reward + gamma * G
            if state not in trajectory[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
