#!/usr/bin/env python3
"""Module for loading the FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the pre-made FrozenLakeEnv from Gymnasium.

    Args:
        desc: None or a list of lists containing a custom map description.
        map_name: None or a string containing a pre-made map name.
        is_slippery: Boolean determining if the ice is slippery.

    Returns:
        The FrozenLake environment.
    """
    return gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                    is_slippery=is_slippery, render_mode="ansi")
