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
    for _ in range(episodes):
        reset_result = env.reset()

        # Gymnasium returns (observation, info).
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result

        states = []
        rewards = []

        for _ in range(max_steps):
            states.append(state)

            action = policy(state)
            result = env.step(action)

            # Gymnasium: observation, reward, terminated, truncated, info
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                # Older Gym API: observation, reward, done, info
                next_state, reward, done, _ = result

            rewards.append(reward)
            state = next_state

            if done:
                break

        # Incremental every-visit Monte Carlo update.
        G = 0
        for i in reversed(range(len(states))):
            G = rewards[i] + gamma * G
            state = states[i]
            V[state] += alpha * (G - V[state])

    return V
