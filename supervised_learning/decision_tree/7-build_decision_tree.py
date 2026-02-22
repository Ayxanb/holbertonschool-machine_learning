#!/usr/bin/env python3
"""
Decision Tree module optimized for vectorized prediction and training.
"""
import numpy as np


def left_child_add_prefix(text):
    """Prefix for non-last child branches."""
    lines = text.splitlines()
    new_text = "+---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "| " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """Prefix for last child branches."""
    lines = text.splitlines()
    new_text = "+---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "  " + line + "\n"
    return new_text


class Node:
    """Internal node class."""
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
        """Finds max depth recursively."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes/leaves recursively."""
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        return (left + right) if only_leaves else (1 + left + right)

    def get_leaves_below(self):
        """Returns list of all leaves below this node."""
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively updates bounds for feature space partitions."""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child, is_left in [(self.left_child, True),
                               (self.right_child, False)]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if is_left:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold
                child.update_bounds_below()

    def update_indicator(self):
        """Creates indicator function for node boundaries."""
        def is_large(x):
            return np.all([x[:, k] > self.lower[k] for k in self.lower],
                          axis=0)

        def is_small(x):
            return np.all([x[:, k] <= self.upper[k] for k in self.upper],
                          axis=0)

        self.indicator = lambda x: np.logical_and(is_large(x), is_small(x))

    def pred(self, x):
        """Recursive prediction for a single data point."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def __str__(self):
        label = "root" if self.is_root else "node"
        out = f"{label} [feature={self.feature}, " \
              f"threshold={self.threshold}]\n"
        out += left_child_add_prefix(self.left_child.__str__())
        out += right_child_add_prefix(self.right_child.__str__())
        return out


class Leaf(Node):
    """Leaf node class."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value, self.is_leaf, self.depth = value, True, depth

    def max_depth_below(self):
        """Finds max depth recursively."""
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
    """Decision Tree model."""
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
        """Prepares the vectorized predict function."""
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
        """Randomly selects a feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            f_slice = self.explanatory[:, feat][node.sub_population]
            f_min, f_max = np.min(f_slice), np.max(f_slice)
            diff = f_max - f_min
        x = self.rng.uniform()
        return feat, (1 - x) * f_min + x * f_max

    def fit(self, explanatory, target, verbose=0):
        """Trains the tree on provided data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory, self.target = explanatory, target
        self.root.sub_population = np.ones_like(target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(explanatory, target)}")

    def fit_node(self, node):
        """Recursive node splitting logic."""
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
            if (node.depth + 1 >= self.max_depth or len(t_sub) <= self.min_pop
                    or np.unique(t_sub).size == 1):
                val = np.bincount(t_sub).argmax()
                setattr(node, side, Leaf(val, node.depth + 1))
            else:
                child = Node(depth=node.depth + 1)
                child.sub_population = pop
                setattr(node, side, child)
                self.fit_node(child)

    def accuracy(self, explanatory, target):
        """Calculates mean accuracy."""
        return np.mean(self.predict(explanatory) == target)

    def __str__(self):
        return self.root.__str__()
