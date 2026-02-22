#!/usr/bin/env python3
"""
Decision Tree printing module.
"""
import numpy as np


def left_child_add_prefix(text):
    """Adds prefix for left child with vertical bar for alignment."""
    lines = text.splitlines()
    new_text = "    +---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "    |      " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """Adds prefix for right child (last child) with empty space."""
    lines = text.splitlines()
    new_text = "    +---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "           " + line + "\n"
    return new_text


class Node:
    """Internal node of the tree."""
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
        
        # Handle threshold formatting to match expected output
        t = self.threshold
        t_str = str(int(t)) if isinstance(t, (int, float)) and t == int(t) \
            else str(t)
            
        out = f"{label} [feature={self.feature}, threshold={t_str}]\n"
        
        if self.left_child:
            out += left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            out += right_child_add_prefix(self.right_child.__str__())
            
        return out.rstrip("\n")


class Leaf(Node):
    """Leaf node of the tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """String representation of a leaf."""
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """Decision tree model container."""
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        """Entry point for printing."""
        return self.root.__str__()
