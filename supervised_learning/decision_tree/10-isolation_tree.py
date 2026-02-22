#!/usr/bin/env python3
"""
Module for Isolation Random Trees used in outlier detection.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Random tree that measures isolation depth for anomaly detection.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """Initializes the tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Matches Decision_Tree string representation."""
        return self.root.__str__()

    def depth(self):
        """Matches Decision_Tree depth calculation."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Matches Decision_Tree node counting."""
        return self.root.count_nodes_below(only_leaves)

    def update_bounds(self):
        """Updates spatial bounds for vectorized prediction."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Returns all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Builds vectorized prediction returning node depth."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                predictions[leaf.indicator(A)] = leaf.depth
            return predictions
        self.predict = predict_func

    def random_split_criterion(self, node):
        """Same random split logic as Decision_Tree."""
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            f_slice = self.explanatory[:, feat][node.sub_population]
            f_min, f_max = np.min(f_slice), np.max(f_slice)
            diff = f_max - f_min
        x = self.rng.uniform()
        return feat, (1 - x) * f_min + x * f_max

    def get_leaf_child(self, node, sub_population):
        """Returns a leaf child with depth as its value."""
        leaf_child = Leaf(value=node.depth + 1, depth=node.depth + 1)
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Returns an internal node child."""
        child = Node(depth=node.depth + 1)
        child.sub_population = sub_population
        return child

    def fit_node(self, node):
        """Recursively splits nodes based on depth and population."""
        node.feature, node.threshold = self.random_split_criterion(node)

        l_mask = self.explanatory[:, node.feature] > node.threshold
        r_mask = ~l_mask
        left_population = np.logical_and(node.sub_population, l_mask)
        right_population = np.logical_and(node.sub_population, r_mask)

        # Is left node a leaf? (Depth limit or single individual)
        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Trains the isolation tree."""
        self.explanatory = explanatory
        # Corrected mask initialization to match dataset size
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
