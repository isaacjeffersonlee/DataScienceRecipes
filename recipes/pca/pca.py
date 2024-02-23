import numpy as np
from typing import Optional


class PCA:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.explained_variance = None
        self.total_var = None
        self.explained_variance_ratio = None

    @staticmethod
    def _svd_flip(u, v):
        """Sign correction to ensure deterministic output from SVD."""
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n, m = X.shape
        if self.n_components is not None:
            if self.n_components > min(n, m):
                raise ValueError("Num components must be between 0 and min(n_samples, n_features)!")
        X = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)  # flip sign of evecs to ensure deterministic output
        # Get variance explained by singular values
        self.explained_variance = (S**2) / (n - 1)
        self.total_var = self.explained_variance.sum()
        self.explained_variance_ratio = self.explained_variance / self.total_var
        return X @ Vt.T[:, :self.n_components]


def main():
    from sklearn.decomposition import PCA as PCA_SK
    from tqdm import trange

    print("Running pca tests...")
    for i in trange(100):
        n, m = np.random.randint(1, 1000), np.random.randint(1, 1000)
        X = np.random.rand(n, m)
        # Note: Sklearn does some weird stuff when we set number of components,
        # so we don't test that.
        pca_sk = PCA_SK()
        pca = PCA()
        Z_sk = pca_sk.fit_transform(X)
        Z = pca.fit_transform(X)
        assert Z.shape == Z_sk.shape
        assert np.allclose(Z, Z_sk)


if __name__ == "__main__":
    main()
