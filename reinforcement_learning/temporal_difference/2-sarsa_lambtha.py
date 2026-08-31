#!/usr/bin/env python3
"""Module that contains the SARSA(lambda) algorithm implementation."""
import numpy as np


def sarsa_lambtha(
    env,
    Q,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05
):
    """Performs the SARSA(lambda) algorithm on an environment.

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q table
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        E = np.zeros_like(Q)

        # Select initial action using epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Select next action using epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1.0

            Q += alpha * delta * E
            E *= gamma * lambtha

            if done:
                break

            state = next_state
            action = next_action

        # Linear decay matching standard assignment spec
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * (
            1 - ep / episodes
        )

    return Q
