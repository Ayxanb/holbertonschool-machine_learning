#!/usr/bin/env python3
"""Module for playing an episode with a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Have the trained agent play an episode, always exploiting Q.

    Args:
        env: The FrozenLakeEnv instance (render_mode="ansi").
        Q: numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        tuple: (total_rewards, rendered_outputs) where rendered_outputs is
            a list of the board states, including the initial and final ones.
    """
    state, _ = env.reset()
    total_rewards = 0.0
    rendered_outputs = [env.render().rstrip('\n')]

    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += float(reward)
        rendered_outputs.append(env.render().rstrip('\n'))
        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
