#!/usr/bin/env python3
"""
Module to define a Decision Tree structure with string representation
"""
import numpy as np


class Node:
    """Represents a node in a decision tree"""
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
        """Recursively finds the maximum depth"""
        left_max = self.left_child.max_depth_below()
        right_max = self.right_child.max_depth_below()
        return max(left_max, right_max)

    def count_nodes_below(self, only_leaves=False):
        """Recursively counts the nodes"""
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        """Returns string representation of the node and its children"""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"

        # Add left child with its specific prefix
        left_str = left_child_add_prefix(self.left_child.__str__())
        # Add right child with its specific prefix
        right_str = right_child_add_prefix(self.right_child.__str__())

        return out + left_str + right_str


class Leaf(Node):
    """Represents a leaf in a decision tree"""
    def __init__(self, value, depth=None):
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

    def __str__(self):
        """Returns string representation of the leaf"""
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """Represents a decision tree model"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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

    def __str__(self):
        """Returns the string representation of the tree"""
        return self.root.__str__()
