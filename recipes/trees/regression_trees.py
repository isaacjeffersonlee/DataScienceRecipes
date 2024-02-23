import numpy as np
from tqdm import tqdm


class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf, feature_bagging=False):
        """
        Initialize DecisionTree object with specified parameters.

        Parameters:
        - max_depth (int): Maximum depth of the decision tree.
        - min_samples_leaf (int): Minimum number of samples required to be in a leaf node.
        - feature_bagging (bool): Flag for feature bagging. If True, randomly samples features during tree building.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_bagging = feature_bagging
        self.tree = {}

    @staticmethod
    def _split(j, s, X, y):
        """
        Split dataset X and labels y based on feature j and threshold s.

        Parameters:
        - j (str): Feature to split on.
        - s (float): Threshold for the split.
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Output labels.

        Returns:
        - list: List containing two dictionaries representing the splits.
        """
        return [{"X": X[idx], "y": y[idx]} for idx in (X[j] < s, X[j] >= s)]

    @staticmethod
    def _loss(R1, R2):
        """
        Calculate the loss for two regions.

        Parameters:
        - R1 (dict): First region.
        - R2 (dict): Second region.

        Returns:
        - float: Total loss for the two regions.
        """
        return ((R1["y"] - R1["y"].mean()) ** 2).sum() + (
            (R2["y"] - R2["y"].mean()) ** 2
        ).sum()

    def _optimal_split(self, X, y):
        """
        Find the optimal split for the given dataset and labels.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Output labels.

        Returns:
        - dict: Dictionary containing information about the optimal split.
        """
        min_loss = np.Inf
        best_R1 = None
        best_R2 = None
        best_j = None
        best_s = None
        features = X.columns
        if self.feature_bagging:
            # Randomly sample features without replacement for feature bagging
            p = len(X.columns)
            features = np.random.choice(features, size=p // 3, replace=False)

        for j in features:
            for s in pd.unique(X[j]):
                R1, R2 = self._split(j, s, X, y)
                l = self._loss(R1, R2)
                if l < min_loss:
                    min_loss = l
                    best_R1 = R1
                    best_R2 = R2
                    best_j = j
                    best_s = s

        return {"R1": best_R1, "R2": best_R2, "j": best_j, "s": best_s}

    def fit(self, X, y):
        """
        Fit the decision tree to the given training data.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Output labels.
        """

        def _recursive_fit(S, d, tree):
            tree["j"] = S["j"]
            tree["s"] = S["s"]
            tree["children"] = [{}, {}]
            for i, Ri in enumerate((S["R1"], S["R2"])):
                num_samples = Ri["X"].shape[0]
                tree["children"][i]["num_samples"] = num_samples
                tree["children"][i]["y_mean"] = Ri["y"].mean()
                tree["children"][i]["parent_y_mean"] = tree[
                    "y_mean"
                ]  # Used for min_samples_leaf edge case
                # Leaf node stop condition
                if num_samples > self.min_samples_leaf and d + 1 < self.max_depth:
                    _recursive_fit(
                        self._optimal_split(Ri["X"], Ri["y"]),
                        d=d + 1,
                        tree=tree["children"][i],
                    )

        _X = X.copy()
        # Initial edge case
        S = self._optimal_split(_X, y)
        self.tree["y_mean"] = _X[S["j"]].mean()
        _recursive_fit(S, d=0, tree=self.tree)

    def predict(self, X):
        """
        Predict the output labels for the given input features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - np.ndarray: Array of predicted output labels.
        """
        assert self.tree, "fit must be called first!"

        def _recursive_predict(x, tree):
            if "j" not in tree:  # Leaf node condition
                # It is possible that we overshoot the min_samples_leaf target and have fewer samples
                # in our leaves. In which case we use the parent node as our new leaf.
                if tree["num_samples"] < self.min_samples_leaf:
                    return tree["parent_y_mean"]
                else:
                    return tree["y_mean"]
            else:
                j, s = tree["j"], tree["s"]
                child_idx = 0 if x[j] < s else 1
                return _recursive_predict(x, tree["children"][child_idx])

        y_pred = np.zeros(X.shape[0])
        for i, (idx, x) in enumerate(X.iterrows()):
            y_pred[i] = _recursive_predict(x, tree=self.tree)

        return y_pred


class DecisionTreeEnsemble:
    def __init__(
        self,
        max_depth,
        min_samples_leaf,
        num_trees,
        feature_bagging=False,
        bootstrap_size=None,
        random_seed=420,
    ):
        """
        Initialize DecisionTreeEnsemble object with specified parameters.

        Parameters:
        - max_depth (int): Maximum depth of individual decision trees in the ensemble.
        - min_samples_leaf (int): Minimum number of samples required to be in a leaf node for individual trees.
        - num_trees (int): Number of decision trees in the ensemble.
        - feature_bagging (bool): Flag for feature bagging in individual trees.
        - bootstrap_size (int, optional): Size of bootstrap samples. If None, it is set to the size of the training set.
        - random_seed (int): Seed for random number generation.
        """
        np.random.seed(random_seed)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_trees = num_trees
        self.bootstrap_size = bootstrap_size
        self.feature_bagging = feature_bagging
        self.forest = []

    def fit(self, X, y):
        """
        Fit the ensemble by training individual decision trees on bootstrap samples.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Output labels.
        """
        N = X.shape[0]
        if self.bootstrap_size is None:
            self.bootstrap_size = N
        # Bootstrap sampling
        for i in range(self.num_trees):
            sample_idx = np.random.choice(
                np.array(range(N)), size=self.bootstrap_size, replace=True
            )
            X_sample = X.iloc[sample_idx, :]
            y_sample = y.iloc[sample_idx]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                feature_bagging=self.feature_bagging,
            )
            tree.fit(X_sample, y_sample)
            self.forest.append(tree)

    def predict(self, X):
        """
        Predict the output labels for the given input features using the ensemble.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - np.ndarray: Array of predicted output labels.
        """
        return np.mean([tree.predict(X) for tree in self.forest], axis=0)


class RandomForest(DecisionTreeEnsemble):
    def __init__(
        self,
        max_depth,
        min_samples_leaf,
        num_trees,
        bootstrap_size=None,
        random_seed=420,
    ):
        """
        Initialize RandomForest object with specified parameters. Inherits from DecisionTreeEnsemble.

        Parameters:
        - max_depth (int): Maximum depth of individual decision trees in the ensemble.
        - min_samples_leaf (int): Minimum number of samples required to be in a leaf node for individual trees.
        - num_trees (int): Number of decision trees in the ensemble.
        - bootstrap_size (int, optional): Size of bootstrap samples. If None, it is set to the size of the training set.
        - random_seed (int): Seed for random number generation.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            num_trees=num_trees,
            feature_bagging=True,
            bootstrap_size=bootstrap_size,
            random_seed=random_seed,
        )


class GradientBoostedDecisionTree:
    def __init__(
        self,
        max_depth,
        min_samples_leaf,
        num_iterations=50,
        learning_rate=0.4,
        show_progress=False,
    ):
        """
        Initialize GradientBoostedDecisionTree object with specified parameters.

        Parameters:
        - max_depth (int): Maximum depth of individual decision trees in the ensemble.
        - min_samples_leaf (int): Minimum number of samples required to be in a leaf node for individual trees.
        - num_iterations (int): Number of boosting iterations.
        - learning_rate (float): Learning rate for each iteration.
        - show_progress (bool): Flag to show progress during fitting.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.show_progress = show_progress
        self.y0 = None
        self.weak_learners = []

    def fit(self, X, y):
        """
        Fit the gradient boosted ensemble to the given training data.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Output labels.
        """
        self.y0 = y_pred = np.mean(y)
        iterations = range(self.num_iterations)
        if self.show_progress:
            iterations = tqdm(iterations, desc="Gradient Boosting")
        for _ in iterations:
            y_resid = -2 * (y - y_pred)
            weak_learner = DecisionTree(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
            )
            weak_learner.fit(X, y_resid)
            self.weak_learners.append(weak_learner)
            y_resid_pred = weak_learner.predict(X)
            y_pred -= self.learning_rate * y_resid_pred

    def predict(self, X):
        """
        Predict the output labels for the given input features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - np.ndarray: Array of predicted output labels.
        """
        assert self.y0 is not None, "fit must be called first!"
        return np.sum(
            [self.y0 * np.ones(X.shape[0])]
            + [
                -1 * self.learning_rate * weak_learner.predict(X)
                for weak_learner in self.weak_learners
            ],
            axis=0,
        )
