import numpy as np
from collections import Counter


def norms(matrix):
    return np.linalg.norm(matrix)


def distance_metric(x1: tuple, x2: tuple):
    H1, W1, Z1 = x1
    H2, W2, Z2 = x2

    # calculate euclidean distance
    res = np.sqrt(
        np.sum(
            [
                (norms(H1) - norms(H2)) ** 2,
                (norms(W1) - norms(W2)) ** 2,
                (norms(Z1) - norms(Z2)) ** 2,
            ]
        )
    )
    return res


class KNN:
    def __init__(self, k=3) -> None:
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X: list, y: list):
        self.X_train += X
        self.y_train += y

    def get(self):
        print(f"K is {self.k}")
        print(f"Total datasets trained {len(self.X_train)}")

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [distance_metric(x, x_train) for x_train in self.X_train]
        k_idxs = np.argsort(distances)[: self.k]

        k_labels = [self.y_train[i] for i in k_idxs]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
