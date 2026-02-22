#!/usr/bin/env python3
"""
Module defining a Decision Tree with vectorized prediction and training.
"""
import numpy as np


def left_child_add_prefix(text):
    """Adds prefix for a child that is NOT the last child."""
    lines = text.splitlines()
    new_text = "+---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "| " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """Adds prefix for the LAST child of a node."""
    lines = text.splitlines()
    new_text = "+---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "  " + line + "\n"
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
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        return (left + right) if only_leaves else (1 + left + right)

    def get_leaves_below(self):
        """Retrieves all leaf nodes in the subtree."""
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Passes bounds down: Left > threshold, Right <= threshold."""
        if self.is_root:
            self.upper, self.lower = {0: np.inf}, {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower, child.upper = self.lower.copy(), self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Creates a vectorized mask function based on feature bounds."""
        def is_large(x):
            return np.all(
                    [x[:, k] > self.lower[k] for k in self.lower], axis=0
                )

        def is_small(x):
            return np.all([x[:, k] <= self.upper[k] for k in self.upper],
                          axis=0)

        self.indicator = lambda x: np.logical_and(is_large(x), is_small(x))

    def pred(self, x):
        """Recursive prediction for a single point."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def __str__(self):
        label = "root" if self.is_root else "node"
        out = f"{label} [feature={self.feature}, threshold={self.threshold}]\n"
        out += left_child_add_prefix(self.left_child.__str__())
        out += right_child_add_prefix(self.right_child.__str__())
        return out


class Leaf(Node):
    """Represents a leaf node."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value, self.is_leaf, self.depth = value, True, depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def pred(self, x):
        return self.value

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision Tree model with random or Gini splitting."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth, self.min_pop = max_depth, min_pop
        self.split_criterion = split_criterion

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def update_predict(self):
        """Builds the vectorized predict function using leaf indicators."""
        self.root.update_bounds_below()
        leaves = self.root.get_leaves_below()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                predictions[leaf.indicator(A)] = leaf.value
            return predictions
        self.predict = predict_func

    def random_split_criterion(self, node):
        """Splits based on a random threshold within feature range."""
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            f_slice = self.explanatory[:, feat][node.sub_population]
            f_min, f_max = np.min(f_slice), np.max(f_slice)
            diff = f_max - f_min
        return feat, \
            (1 - self.rng.uniform()) * f_min + self.rng.uniform() * f_max

    def fit(self, explanatory, target, verbose=0):
        """Fits the model to the training data."""
        self.split_criterion = self.random_split_criterion \
            if self.split_criterion == "random" else self.Gini_split_criterion
        self.explanatory, self.target = explanatory, target
        self.root.sub_population = np.ones_like(target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            msg = ("  Training finished.\n"
                   "    - Depth                     : {}\n"
                   "    - Number of nodes           : {}\n"
                   "    - Number of leaves          : {}\n"
                   "    - Accuracy on training data : {}")
            print(msg.format(self.depth(), self.count_nodes(),
                             self.count_nodes(True),
                             self.accuracy(explanatory, target)))

    def fit_node(self, node):
        """Recursive training: creates children or marks leaves."""
        targets = self.target[node.sub_population]
        if (node.depth >= self.max_depth or len(targets) <= self.min_pop or
                np.unique(targets).size == 1):
            return

        node.feature, node.threshold = self.split_criterion(node)
        l_mask = self.explanatory[:, node.feature] > node.threshold
        r_mask = ~l_mask
        l_pop = np.logical_and(node.sub_population, l_mask)
        r_pop = np.logical_and(node.sub_population, r_mask)

        if not np.any(l_pop) or not np.any(r_pop):
            return

        for side, pop in [('left_child', l_pop), ('right_child', r_pop)]:
            t_sub = self.target[pop]
            if (node.depth + 1 >= self.max_depth or
                    len(t_sub) <= self.min_pop or
                    np.unique(t_sub).size == 1):
                val = np.bincount(t_sub).argmax()
                setattr(node, side, Leaf(val, node.depth + 1))
            else:
                child = Node(depth=node.depth + 1)
                child.sub_population = pop
                setattr(node, side, child)
                self.fit_node(child)

    def accuracy(self, explanatory, target):
        return np.mean(self.predict(explanatory) == target)

    def __str__(self):
        return self.root.__str__()
