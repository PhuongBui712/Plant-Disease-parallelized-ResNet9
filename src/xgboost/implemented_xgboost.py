import numpy as np
from xgboost.decision_tree import DecisionTree


class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        m, n = X.shape
        self.y_mean = np.mean(y)
        F_m = np.full(m, self.y_mean)

        for _ in range(self.n_estimators):
            residuals = y - F_m
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F_m += self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        # TODO: Trung
        F_m = np.full(X.shape[0], self.y_mean)
        for tree in self.trees:
            F_m += self.learning_rate * tree.predict(X)
        return F_m
