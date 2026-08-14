#!/usr/bin/env python3

"""Module for playing an episode using a trained Q-learning agent."""

import numpy as np


def play(env, Q, max_steps=100):
    """Have a trained agent play an episode using a greedy policy.

    Args:
        env: The FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        tuple: (total_reward, rendered_outputs) containing total reward
            and a list of board states rendered as ANSI strings.
    """
    rendered_outputs = []
    total_reward = 0

    reset_res = env.reset()
    state = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    board = env.render()
    rendered_outputs.append(board)
    print(board, end="")

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        step_res = env.step(action)

        if len(step_res) == 5:
            next_state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_res

        total_reward += reward
        board = env.render()
        rendered_outputs.append(board)
        print(board, end="")

        state = next_state

        if done:
            break

    return total_reward, rendered_outputs
