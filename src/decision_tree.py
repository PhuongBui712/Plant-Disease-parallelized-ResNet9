import numpy as np
from numba import njit, prange


@njit
def unique_counts(arr):
    """Calculate unique values and their counts in an array."""
    unique_vals = []
    counts = []
    for val in arr:
        if val in unique_vals:
            counts[unique_vals.index(val)] += 1
        else:
            unique_vals.append(val)
            counts.append(1)
    return np.array(unique_vals), np.array(counts)


@njit
def entropy(y):
    """Calculate the entropy of a dataset."""
    _, counts = unique_counts(y)
    norm_counts = counts / counts.sum()
    return -np.sum(norm_counts * np.log2(norm_counts + 1e-9))  # Adding a small constant to avoid log(0)


@njit(parallel=True)
def parallel_split(X_column, split_thresh):
    left_idxs = np.where(X_column <= split_thresh)[0]
    right_idxs = np.where(X_column > split_thresh)[0]
    return left_idxs, right_idxs

# @njit(parallel=True)
@njit
def information_gain(y, X_column, split_thresh):
    parent_entropy = entropy(y)
    left_idxs, right_idxs = parallel_split(X_column, split_thresh)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

    ig = parent_entropy - child_entropy
    return ig


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.arange(num_features)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = parallel_split(X[:, best_feat], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _most_common_label(self, y):
        unique_vals, counts = unique_counts(y)
        return unique_vals[np.argmax(counts)]

    def _predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

