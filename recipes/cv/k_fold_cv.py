from multiprocessing import Pool
import numpy as np


def eval_fold(model, metrics, X_train_cv, y_train_cv, X_test_cv, y_test_cv):
    """
    Evaluate a model on a fold.

    Parameters:
    - model: The machine learning model to evaluate.
    - metrics: Dictionary of evaluation metrics.
    - X_train_cv: Training features for the current fold.
    - y_train_cv: Training labels for the current fold.
    - X_test_cv: Testing features for the current fold.
    - y_test_cv: Testing labels for the current fold.

    Returns:
    - dict: Evaluation results for each metric.
    """
    model.fit(X_train_cv, y_train_cv)
    y_test_cv_pred = model.predict(X_test_cv)
    return {metric: func(y_test_cv, y_test_cv_pred) for metric, func in metrics.items()}


def k_fold_cv(k, model, X, y, metrics, random_seed=100):
    """
    Perform k-fold cross-validation with parallelized evaluation.

    Parameters:
    - k: Number of folds.
    - model: The machine learning model to cross-validate.
    - X: Input features.
    - y: Output labels.
    - metrics: Dictionary of evaluation metrics.
    - random_seed: Seed for randomization.

    Returns:
    - dict: Evaluation results for each metric.
    """
    N = X.shape[0]
    idx = np.array(range(N))
    np.random.seed(random_seed)
    np.random.shuffle(idx)  # Shuffles inplace
    partition_size = N // k
    partitions = [
        idx[i * partition_size : min((i + 1) * partition_size, N)] for i in range(k)
    ]
    assert len(partitions) == k, "Number of partitions does not match k."

    train_idx = [
        np.concatenate([partitions[i] for i in range(k) if i != test_fold_index])
        for test_fold_index in range(k)
    ]
    test_idx = partitions
    X_train_cv = [X.iloc[idx, :] for idx in train_idx]
    y_train_cv = [y.iloc[idx] for idx in train_idx]
    X_test_cv = [X.iloc[idx, :] for idx in test_idx]
    y_test_cv = [y.iloc[idx] for idx in test_idx]

    # Parallelize evaluation over folds
    with Pool() as pool:
        results = pool.starmap(
            eval_fold,
            [
                (
                    model,
                    metrics,
                    X_train_cv[i],
                    y_train_cv[i],
                    X_test_cv[i],
                    y_test_cv[i],
                )
                for i in range(k)
            ],
        )

    return {metric: [result[metric] for result in results] for metric in metrics}
