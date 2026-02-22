#!/usr/bin/env python3
"""
Decision Tree module with recursive string representation.
"""
import numpy as np


def left_child_add_prefix(text):
    """
    Adds prefixes for left children.
    The first line gets the arrow, others get the vertical bar.
    """
    lines = text.splitlines()
    new_text = "+---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "|      " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Adds prefixes for right children.
    The first line gets the arrow, others get empty space.
    """
    lines = text.splitlines()
    new_text = "+---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "       " + line + "\n"
    return new_text


class Node:
    """Represents an internal node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Recursively finds the maximum depth."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Recursively counts the nodes."""
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        """Recursive string representation to match specific stdout."""
        label = "root" if self.is_root else "node"
        # Format threshold: integers as int, floats as float
        t_val = self.threshold
        if isinstance(t_val, (int, float)) and t_val == int(t_val):
            t_str = str(int(t_val))
        else:
            t_str = str(t_val)

        out = f"{label} [feature={self.feature}, threshold={t_str}]\n"

        if self.left_child is not None:
            out += left_child_add_prefix(self.left_child.__str__())

        if self.right_child is not None:
            out += right_child_add_prefix(self.right_child.__str__())

        return out.rstrip("\n")


class Leaf(Node):
    """Represents a leaf in a decision tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 for the leaf."""
        return 1

    def __str__(self):
        """Leaf string."""
        return f"leaf [value={self.value}]"


class Decision_Tree():
    """Represents a decision tree model."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """Returns the maximum depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts nodes starting from root."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Entry point for tree string representation."""
        return self.root.__str__()
