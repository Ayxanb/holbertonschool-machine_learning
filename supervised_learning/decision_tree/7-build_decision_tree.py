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

    def update_indicator(self):
        """
        Computes the indicator function from the lower and upper bounds
        and stores it in the attribute self.indicator.
        """
        def is_large_enough(x):
            """
            Returns a boolean array where True indicates the individual's
            features are all greater than the lower bounds.
            """
            # Create a list of boolean arrays for each feature in lower
            checks = [np.greater(x[:, key], self.lower[key])
                      for key in self.lower.keys()]
            # Combine them: True only if ALL features satisfy the condition
            return np.all(checks, axis=0)

        def is_small_enough(x):
            """
            Returns a boolean array where True indicates the individual's
            features are all less than or equal to the upper bounds.
            """
            # Create a list of boolean arrays for each feature in upper
            checks = [np.less_equal(x[:, key], self.upper[key])
                      for key in self.upper.keys()]
            # Combine them: True only if ALL features satisfy the condition
            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Recursively traverses the tree for a single individual x"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

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

    def pred(self, x):
        """Returns the value of the leaf for a single individual x"""
        return self.value

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

    def update_predict(self):
        """
        Computes the prediction function based on leaf indicators.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_function(A):
            """
            Evaluates the prediction for a 2D array A.
            """
            # Initialize results array with zeros of shape (n_individuals,)
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                # Get boolean mask for individuals in this leaf
                mask = leaf.indicator(A)
                # Assign the leaf's value to those individuals
                predictions[mask] = leaf.value
            return predictions

        self.predict = predict_function

    def pred(self, x):
        """Entry point for the recursive prediction on a single individual x"""
        return self.root.pred(x)

    def fit_node(self, node):
        """
        Recursively trains the decision tree by splitting nodes
        until leaf conditions are met.
        """
        # Determine the target values for the current node's population
        node_targets = self.target[node.sub_population]

        # 1. Check Leaf Conditions:
        # - Max depth reached
        # - Population too small
        # - Node is pure (all targets are the same)
        if (node.depth >= self.max_depth or
                len(node_targets) <= self.min_pop or
                np.unique(node_targets).size == 1):
            # Transform current node into a leaf by changing its children logic
            # In this structure, we replace it or call a leaf generator
            return

        # 2. Split the data:
        node.feature, node.threshold = self.split_criterion(node)

        # Create population masks (Boolean arrays)
        # Left child = feature > threshold
        left_mask = self.explanatory[:, node.feature] > node.threshold
        right_mask = self.explanatory[:, node.feature] <= node.threshold

        # Combine with parent's sub_population to get absolute indices
        left_population = np.logical_and(node.sub_population, left_mask)
        right_population = np.logical_and(node.sub_population, right_mask)

        # 3. Handle Child Creation:
        # Check if children would be empty
        # (if so, this node is effectively a leaf)
        if not np.any(left_population) or not np.any(right_population):
            return

        # Create Left Child
        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Create Right Child
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf and assigns it the majority class value."""
        # Find the most frequent target value in this sub_population
        targets = self.target[sub_population]
        value = np.bincount(targets).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node for further splitting."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit(self, explanatory, target, verbose=0):
        """
        Trains the decision tree using the specified split criterion.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            # Gini_split_criterion to be defined in future tasks
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.\n"
                  f"- Depth                     : {self.depth()}\n"
                  f"- Number of nodes           : {self.count_nodes()}\n"
                  f"- Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}\n"
                  f"- Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """
        Recursively splits the node until leaf conditions are met.
        """
        # Determine if current node targets are pure
        node_targets = self.target[node.sub_population]

        # Stop conditions: max depth, min population, or pure node
        if (node.depth >= self.max_depth or
                len(node_targets) <= self.min_pop or
                np.unique(node_targets).size == 1):
            # This node is handled by parent as a leaf already
            return

        node.feature, node.threshold = self.split_criterion(node)

        # Splitting logic: Left (> threshold), Right (<= threshold)
        left_mask = self.explanatory[:, node.feature] > node.threshold
        right_mask = self.explanatory[:, node.feature] <= node.threshold

        left_pop = np.logical_and(node.sub_population, left_mask)
        right_pop = np.logical_and(node.sub_population, right_mask)

        # Handle Left Child
        if (np.sum(left_pop) <= self.min_pop or
                node.depth + 1 >= self.max_depth or
                np.unique(self.target[left_pop]).size == 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        # Handle Right Child
        if (np.sum(right_pop) <= self.min_pop or
                node.depth + 1 >= self.max_depth or
                np.unique(self.target[right_pop]).size == 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates and returns a Leaf child."""
        # Find majority class
        targets = self.target[sub_population]
        value = np.bincount(targets).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates and returns an internal Node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates accuracy on a given dataset."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def np_extrema(self, arr):
        """Returns the minimum and maximum of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Determines a random feature and threshold for splitting"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            # Placeholder for Gini (prevents error if called)
            self.split_criterion = getattr(self, "Gini_split_criterion",
                                           self.random_split_criterion)

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"  Training finished.\n"
                f"    - Depth                     \
                        : {self.depth()}\n"
                f"    - Number of nodes           \
                        : {self.count_nodes()}\n"
                f"    - Number of leaves          \
                        : {self.count_nodes(True)}\n"
                f"    - Accuracy on training data \
                        : {self.accuracy(self.explanatory, self.target)}"
            )

    def __str__(self):
        """Entry point for tree string representation"""
        return self.root.__str__()
