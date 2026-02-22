#!/usr/bin/env python3
"""
Module to define a Decision Tree structure with custom string representation
"""
import numpy as np


def left_child_add_prefix(text):
    """
    Adds prefix for a child that is NOT the last child.
    The pipe '|' must continue down to connect to the next sibling.
    """
    lines = text.splitlines()
    # First line gets the branch arrow
    new_text = "+---> " + lines[0] + "\n"
    # Following lines get a vertical pipe to continue the line
    for line in lines[1:]:
        new_text += "| " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Adds prefix for the LAST child of a node.
    No pipe is needed below this branch.
    """
    lines = text.splitlines()
    # First line gets the branch arrow
    new_text = "+---> " + lines[0] + "\n"
    # Following lines get empty spaces
    for line in lines[1:]:
        new_text += "  " + line + "\n"
    return new_text


class Node:
    """Represents a node in a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Recursively finds the maximum depth"""
        left_max = self.left_child.max_depth_below()
        right_max = self.right_child.max_depth_below()
        return max(left_max, right_max)

    def count_nodes_below(self, only_leaves=False):
        """Recursively counts the nodes"""
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def get_leaves_below(self):
        """
        Recursively retrieves all leaf nodes in the subtree
        """
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """
        Recursively compute the lower and upper bounds for each feature.
        Note: Left child = Greater than threshold;
        Right child = Less than or equal to threshold.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        self.left_child.lower = self.lower.copy()
        self.left_child.upper = self.upper.copy()
        self.left_child.lower[self.feature] = self.threshold

        self.right_child.lower = self.lower.copy()
        self.right_child.upper = self.upper.copy()
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def __str__(self):
        """Returns string representation matching the checker's format"""
        if self.is_root:
            out = \
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = \
                f"node [feature={self.feature}, threshold={self.threshold}]\n"

        out += left_child_add_prefix(self.left_child.__str__())
        out += right_child_add_prefix(self.right_child.__str__())

        return out


class Leaf(Node):
    """Represents a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        """Initializes a leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 for the leaf"""
        return 1

    def get_leaves_below(self):
        """Returns the leaf itself in a list"""
        return [self]

    def update_bounds_below(self):
        """Base case: leaves do not have children to update"""
        pass

    def __str__(self):
        """
        Returns string representation of the leaf
        matching the desired output format.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """Represents a decision tree model"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the decision tree"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """Returns the maximum depth"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts nodes starting from root"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Entry point for the checker to retrieve all leaves"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Entry point to calculate bounds for the entire tree"""
        self.root.update_bounds_below()

    def __str__(self):
        """Entry point for tree string representation"""
        return self.root.__str__()
