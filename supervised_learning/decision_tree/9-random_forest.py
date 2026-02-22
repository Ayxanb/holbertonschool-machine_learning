#!/usr/bin/env python3
"""
Module defining a Random Forest ensemble model.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """
    Represents a random forest of multiple decision trees.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the random forest.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts classes by taking the mode of all tree predictions.

        Args:
            explanatory (numpy.ndarray): 2D array of features.

        Returns:
            numpy.ndarray: 1D array of predicted classes.
        """
        # Collect predictions from each tree: results in shape (n_trees, n)
        all_tree_preds = np.array([p(explanatory) for p in self.numpy_preds])

        # To find the mode without complex loops, we can use bincount per column
        # or simply transpose and apply a custom mode function.
        # Here we use a robust method to find the mode along axis 0.
        def get_mode(column):
            return np.bincount(column.astype(int)).argmax()

        return np.apply_along_axis(get_mode, 0, all_tree_preds)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the forest by fitting multiple decision trees.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            # Each tree gets a unique seed derived from the forest seed
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)

            # Store the vectorized predict function of the tree
            self.numpy_preds.append(T.predict)

            # Collect metrics for verbose output
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            # Training stats formatting
            print("  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")
            print(f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the forest on a given dataset.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
