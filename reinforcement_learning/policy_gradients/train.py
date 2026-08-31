#!/usr/bin/env python3
"""Train a policy-gradient agent."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train a policy-gradient agent in an environment.

    Args:
        env: Initial environment.
        nb_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        show_result: Render the environment every 1000 episodes when True.

    Returns:
        A numpy array containing the score from each episode.
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        state = np.asarray(state)
        gradients = []
        rewards = []
        done = False

        while not done:
            action, grad = policy_gradient(state, weight)
            result = env.step(action)

            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            gradients.append(grad)
            rewards.append(reward)
            state = np.asarray(next_state)

            if show_result and (episode + 1) % 1000 == 0:
                env.render()

        score = sum(rewards)
        scores.append(score)

        returns = []
        discounted = 0
        for reward in reversed(rewards):
            discounted = reward + gamma * discounted
            returns.append(discounted)
        returns.reverse()

        for grad, reward_to_go in zip(gradients, returns):
            weight += alpha * reward_to_go * grad

        print("Episode: {} Score: {}".format(episode + 1, score))

    return np.asarray(scores)
