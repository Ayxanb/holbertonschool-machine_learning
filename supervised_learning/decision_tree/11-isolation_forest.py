#!/usr/bin/env python3
"""
Module defining the Isolation Random Forest for outlier detection.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Ensemble of isolation trees to detect outliers via mean depth.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the forest.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Calculates the mean depth for each individual across all trees.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fits n_trees isolation trees to the data.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print("  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows with the smallest mean depth.

        Args:
            explanatory (numpy.ndarray): The dataset to check.
            n_suspects (int): Number of outliers to return.

        Returns:
            tuple: (suspect_rows, depths_of_suspects)
        """
        # Calculate mean depths for all rows
        mean_depths = self.predict(explanatory)

        # Get indices of the n smallest depths (outliers)
        # argsort returns indices that would sort the array
        indices = np.argsort(mean_depths)[:n_suspects]

        # Return the actual rows and their corresponding depths
        return explanatory[indices], mean_depths[indices]
