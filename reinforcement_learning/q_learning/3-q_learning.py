#!/usr/bin/env python3

"""Module for training a Q-learning agent on Gymnasium environments."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning on a given FrozenLake environment.

    Args:
        env: The FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial threshold for epsilon-greedy action selection.
        min_epsilon: Minimum value that epsilon should decay to.
        epsilon_decay: Decay rate for updating epsilon between episodes.

    Returns:
        tuple: (Q, total_rewards) where Q is the updated Q-table and
            total_rewards is a list containing the total reward per episode.
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        reset_res = env.reset()
        state = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            step_res = env.step(action)

            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res

            if done and reward == 0:
                reward = -1.0

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q, total_rewards
