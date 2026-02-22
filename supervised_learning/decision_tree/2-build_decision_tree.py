#!/usr/bin/env python3
"""
Decision Tree module with precise string formatting.
"""
import numpy as np


def left_child_add_prefix(text):
    """Adds prefixes for left children with vertical bars."""
    lines = text.splitlines()
    # First line gets the arrow
    new_text = "    +---> " + lines[0] + "\n"
    # Subsequent lines get the bar aligned under the '+'
    for line in lines[1:]:
        new_text += "    |     " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """Adds prefixes for right children with empty space."""
    lines = text.splitlines()
    # First line gets the arrow
    new_text = "    +---> " + lines[0] + "\n"
    # Subsequent lines get empty space
    for line in lines[1:]:
        new_text += "          " + line + "\n"
    return new_text


class Node:
    """Node class for Decision Tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def __str__(self):
        """Recursive string representation."""
        label = "root" if self.is_root else "node"

        # Match threshold formatting (30000.0 -> 30000)
        t = self.threshold
        if isinstance(t, (int, float)) and t == int(t):
            t_str = str(int(t))
        else:
            t_str = str(t)

        out = f"{label} [feature={self.feature}, threshold={t_str}]\n"

        if self.left_child is not None:
            out += left_child_add_prefix(self.left_child.__str__())

        if self.right_child is not None:
            out += right_child_add_prefix(self.right_child.__str__())

        return out.rstrip("\n")


class Leaf(Node):
    """Leaf class for Decision Tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Returns the leaf string label."""
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """Decision Tree class wrapper."""
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        """Returns the string of the root node."""
        return self.root.__str__()
