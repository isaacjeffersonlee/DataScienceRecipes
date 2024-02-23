# An implementation of lloyds algorithm
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


class KMeansClassifier:
    def __init__(self, k: int):
        self.k = k
        self.centroids = None

    def _get_init_centroids(self, X: np.ndarray) -> None:
        """k-means++ recursive implementation for finding initial centroids.

        For more info: https://en.wikipedia.org/wiki/K-means%2B%2B

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            The data to partition, where each row of X
            represents a vector/observation to classify.
            Note: We assume X has no missing values.
        k : int
            Number of clusters.
        """
        X = np.unique(X, axis=0)
        N_0 = X.shape[0]

        def _get_centroids(
            X: np.ndarray, p: np.ndarray, centroids: list[np.ndarray] = []
        ) -> list[np.ndarray]:
            N = X.shape[0]
            if N == N_0 - self.k:
                return centroids
            else:
                centroid_row_idx = np.random.choice(N, size=1, p=p)
                centroid = X[centroid_row_idx, :]
                X_without_centroid = np.delete(X, centroid_row_idx, axis=0)
                ssd = np.sqrt(np.sum((X_without_centroid - centroid) ** 2, axis=1))
                ssd = (ssd - np.min(ssd)) / (
                    np.max(ssd) - np.min(ssd)
                )  # Normalize to avoid overflow errors
                p = softmax(ssd**2)
                centroids.append(centroid)
                return _get_centroids(X_without_centroid, p, centroids)

        p0 = np.ones(N_0) / N_0  # Initial uniform distribution
        self.centroids = _get_centroids(X, p=p0)

    def predict(self, X: np.ndarray, centroids: np.ndarray = None) -> np.ndarray:
        """
        Predict the class labels of the rows of X using the kmeans classifier.

        For each row in X (each observation), the nearest centroid is found
        and then the prediction for that row corresponds to the label assigned
        to that centroid.

        Parameters
        ----------
        X : np.ndarray
            Data to predict, with each row representing an observation and
            each column representing a variable/feature.

        centroids : np.ndarray
            Optionally specify the centroids to use for prediction.
            This is useful if we are loading the centroids from a pkl file
            and we don't want to have to re-fit the data everytime we
            want to use the classifier.

        Returns
        -------
        np.ndarray, shape (N,)
            A 1D numpy array of predicted class labels.
        """
        if centroids is None:
            centroids = self.centroids

        assert centroids is not None, (
            "No data has been fitted!,"
            " Please use the fit method to first"
            "fit the classifier before using the predict method."
        )
        # Fix the order of the centroids
        centroids = np.sort(centroids, axis=0)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))  # Add second dimension
        N_0, d = X.shape
        # sum of squared distances
        ssd = (X - np.array(centroids)) ** 2
        # assert k == ssd.shape[0]
        # assert (N_0, d) == ssd[0].shape
        ssd = ssd.reshape((self.k * N_0, d))
        ssd = np.sqrt(np.sum(ssd, axis=1))
        ssd = ssd.reshape((self.k, N_0)).T
        return np.argmin(ssd, axis=1)

    def fit(self, X: np.ndarray, max_iters: int = 100) -> np.ndarray:
        """Perform the k-means clustering algorithm.

        Uses k-means++ algorithm to initialize centroids
        then Lloyds' algorithm to partition N data
        observations into k samples.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            The data to partition, where each row of X
            represents a vector/observation to classify.
            Note: We drop any rows of X that have missing values.

        Returns
        -------
        np.ndarray, shape (N,)
            1D array of classification
            labels, where the ith element is an integer label,
            classifying which cluster the ith observation,
            (i.e ith row of X) belongs to.
        """
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))  # Add second dimension

        X = X[~np.isnan(X).any(axis=1), :]  # Remove rows with missing values
        self._get_init_centroids(X)  # Get initial centroids using k-means++
        for i in range(max_iters):  # TODO: Add convergence metric
            labels = self.predict(X)
            new_centroids = []
            for idx, centroid in enumerate(list(self.centroids)):
                mask = np.asarray(labels == idx)
                X_in_cluster = X[mask, :]
                centroid = np.mean(X_in_cluster, axis=0).reshape((1, -1))
                new_centroids.append(centroid)

            self.centroids = new_centroids  # Overwrite previous iterations value

        return labels

    def save_centroids(self, file: str) -> None:
        """Save self.centroids to a .npy file at file path."""
        assert self.centroids is not None, (
            "No data has been fitted!,"
            " Please use the fit method to first"
            "fit the classifier before using the save_centroids method."
        )
        with open(file, "wb") as f:
            np.save(f, np.array(self.centroids), allow_pickle=True)

    def load_centroids(self, file: str) -> np.ndarray:
        """Load centroids from a .npy file at file path."""
        with open(file, "rb") as f:
            centroids = np.load(f, allow_pickle=True)
        return centroids


def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pca

    df = pd.read_csv("../Data/iris.csv", skiprows=1, names=["feat1", "feat2", "class"])
    df = df.reset_index()
    true_labels = np.array(df["class"])
    df = df.drop(columns=["class"])
    X = np.array(df)
    Z, explained_variance = pca.get_pc(X, 2)
    clf = KMeansClassifier(k=3)
    labels = clf.fit(X, max_iters=20)
    X_unseen = np.array([5, 5, 5, 5])
    unseen_labels = clf.predict(X_unseen)
    print("Example unseen data prediction: ", unseen_labels)

    z1, z2 = Z.T
    fig, ax = plt.subplots(nrows=2, ncols=1)
    g1 = sns.scatterplot(x=z1, y=z2, hue=labels, ax=ax[0])
    g2 = sns.scatterplot(x=z1, y=z2, hue=true_labels, ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
