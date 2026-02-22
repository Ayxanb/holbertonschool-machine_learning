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
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        """
        Initializes a leaf node.

        Args:
            value: The predicted class or value for this leaf.
            depth: The depth level of the leaf within the tree.
        """
        super().__init__()
        self.value, self.is_leaf, self.depth = value, True, depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf as the base case for recursion.

        Returns:
            int: The depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts this leaf for the tree node summation.

        Args:
            only_leaves (bool): If True, counts only leaves.

        Returns:
            int: 1, representing this single leaf.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf.

        Returns:
            list: A list containing the current Leaf instance.
        """
        return [self]

    def update_bounds_below(self):
        """
        Base case for bound updates; leaves have no children to update.
        """
        pass

    def pred(self, x):
        """
        Returns the prediction value for a given individual.

        Args:
            x (numpy.ndarray): A 1D array representing a single individual.

        Returns:
            The value stored in the leaf.
        """
        return self.value

    def __str__(self):
        """
        Returns the string representation of the leaf.

        Returns:
            str: Formatted string indicating leaf value.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    A decision tree classifier that supports random and Gini splitting.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes the Decision Tree.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_pop (int): Minimum number of individuals to split a node.
            seed (int): Seed for the random number generator.
            split_criterion (str): Method to choose splits ("random", "Gini").
            root (Node): Root node of the tree.
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth, self.min_pop = max_depth, min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
            int: Maximum depth level.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes in the tree.

        Args:
            only_leaves (bool): If True, only counts leaf nodes.

        Returns:
            int: Total count of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves)

    def update_predict(self):
        """
        Builds a vectorized prediction function based on leaf indicators.
        """
        self.root.update_bounds_below()
        leaves = self.root.get_leaves_below()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """Vectorized prediction internal function."""
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                predictions[leaf.indicator(A)] = leaf.value
            return predictions
        self.predict = predict_func

    def random_split_criterion(self, node):
        """
        Chooses a random feature and threshold for splitting.

        Args:
            node (Node): The node to be split.

        Returns:
            tuple: (feature_index, threshold)
        """
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            f_slice = self.explanatory[:, feat][node.sub_population]
            f_min, f_max = np.min(f_slice), np.max(f_slice)
            diff = f_max - f_min
        x = self.rng.uniform()
        return feat, (1 - x) * f_min + x * f_max

    def fit(self, explanatory, target, verbose=0):
        """
        Trains the tree using the provided features and targets.

        Args:
            explanatory (numpy.ndarray): 2D array of features.
            target (numpy.ndarray): 1D array of target labels.
            verbose (int): If 1, prints training statistics.
        """
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
        """
        Recursively builds children for a node until leaf criteria are met.

        Args:
            node (Node): The node to process.
        """
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
        """
        Calculates the classification accuracy.

        Args:
            explanatory (numpy.ndarray): 2D array of features.
            target (numpy.ndarray): 1D array of target labels.

        Returns:
            float: Percentage of correct predictions.
        """
        return np.mean(self.predict(explanatory) == target)

    def __str__(self):
        """
        Returns string representation of the root node.

        Returns:
            str: The tree structure as a string.
        """
        return self.root.__str__()
