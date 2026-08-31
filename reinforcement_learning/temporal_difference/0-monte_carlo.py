import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

    env      - environment instance
    V        - numpy.ndarray of shape (s,), the value estimate
    policy   - function that takes a state and returns the next action
    episodes - total number of episodes to train over
    max_steps - maximum number of steps per episode
    alpha    - learning rate
    gamma    - discount rate

    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        state = env.reset()[0]
        episode = []

        # Generate an episode
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        episode = np.array(episode, dtype=int)
        G = 0

        # Work backwards through the episode, computing returns
        for t in reversed(range(len(episode))):
            state_t, reward_t = episode[t]
            G = reward_t + gamma * G

            # First-visit check: only update if this state hasn't
            # appeared earlier in the episode
            if state_t not in episode[:t, 0]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
