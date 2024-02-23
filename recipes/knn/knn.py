import numpy as np


class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.X_train = None
        self.y_train = None

    @staticmethod
    def _distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))  # Euclidean distance

    def _get_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        n, r1 = X1.shape
        m, r2 = X2.shape
        assert r1 == r2, "X1 and X2 must have the same number of columns!"
        d = np.zeros((n, m))
        if n > m:
            for j in range(m):
                d[:, j] = self._distance(X1, X2[j])
        else:
            for i in range(n):
                d[i, :] = self._distance(X1[i], X2)

        return d

    def _get_k_neighbors_idx(self, X: np.ndarray) -> np.ndarray:
        distances = self._get_distances(X, self.X_train)
        # Get indices of k smallest valued columns, for each row
        return np.argpartition(distances, kth=self.k, axis=1)[:, : self.k], distances

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.X_train is not None, "fit must be called first!"
        X = np.array(X)
        k_neighbors_idx, _ = self._get_k_neighbors_idx(X)
        y_k_neighbors = self.y_train[k_neighbors_idx]
        # Get the mode of the neighbors' classes
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=y_k_neighbors
        )


class DistanceWeightedKNNClassifier(KNNClassifier):
    def predict(self, X):
        assert self.X_train is not None, "fit must be called first!"
        X = np.array(X)
        k_neighbors_idx, distances = self._get_k_neighbors_idx(X)
        y_k_neighbors = self.y_train[k_neighbors_idx]
        N = X.shape[0]
        # Deal with zero division errors
        with np.errstate(all="ignore"):
            w = 1 / np.array([distances[i, k_neighbors_idx[i, :]] for i in range(N)])
            w[w == np.inf] = np.max(w)

        return np.array(
            [
                np.bincount(y_k_neighbors[i, :], weights=w[i, :]).argmax()
                for i in range(N)
            ]
        )
