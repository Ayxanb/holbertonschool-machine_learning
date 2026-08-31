#!/usr/bin/env python3
"""SARSA(lambda) algorithm using eligibility traces."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose the next action using the epsilon-greedy policy.

    Args:
        Q: numpy.ndarray containing the Q table.
        state: the current state.
        epsilon: the epsilon to use for the calculation.

    Returns:
        The next action index.
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm.

    Args:
        env: the environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: the eligibility trace factor.
        episodes: the total number of episodes to train over.
        max_steps: the maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.
        epsilon: the initial threshold for epsilon greedy.
        min_epsilon: the minimum value that epsilon should decay to.
        epsilon_decay: the decay rate for updating epsilon between
            episodes.

    Returns:
        Q, the updated Q table.
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(
                action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action]
            delta -= Q[state, action]

            eligibility[state, action] += 1

            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

    return Q
