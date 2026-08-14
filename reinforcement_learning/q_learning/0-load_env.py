"""Module for initializing Gymnasium reinforcement learning environments."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLake-v1 environment from Gymnasium.

    Args:
        desc: Custom map description or None.
        map_name: Pre-made map name ('4x4', '8x8') or None.
        is_slippery: Boolean indicating if ice actions are stochastic.

    Returns:
        The instantiated Gymnasium environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi",
    )
    return env
