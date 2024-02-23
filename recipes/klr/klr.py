import numpy as np

# TODO: Classify with sk-learn like API

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def laplacian_kernel(x, y, gamma):
    assert gamma > 0
    return np.exp(-gamma * np.sum(np.abs(x - y), axis=0))


def kernel_matrix(X1, X2, gamma):
    N1, N2 = X1.shape[0], X2.shape[0]
    K = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            K[i, j] = laplacian_kernel(X1[i], X2[j], gamma)

    return K


def propagate(X, y, alpha, lamda, K):
    y_log = predict_log(X, alpha, K)
    # Zero out final element of alpha since regularization term does not include intercept
    alpha_ = alpha.copy().reshape(-1, 1)
    alpha_[-1] = 0
    mean_loss = (
        -np.mean(y * np.log(y_log) + (1 - y) * np.log(1 - y_log))
        + (lamda / 2) * alpha_.T @ K @ alpha_
    )
    dalpha = -np.mean(K * (y - y_log), axis=1).reshape(-1, 1) + lamda * K @ alpha_
    mean_loss = np.squeeze(mean_loss)
    grads = {"dalpha": dalpha}
    return grads, mean_loss


def optimise(
    X, y, alpha, lamda, K, num_iterations=1000, learning_rate=0.1, print_loss=False
):
    mean_loss_history = []
    for i in range(num_iterations):
        grads, mean_loss = propagate(X, y, alpha, lamda, K)
        dalpha = grads["dalpha"]
        alpha = alpha - learning_rate * dalpha
        if i % 100 == 0:
            mean_loss_history.append(mean_loss)

        if print_loss and i % 100 == 0:
            print(f"Mean loss after iteration {i}: {mean_loss}")

    params = {"alpha": alpha}
    grads = {"dalpha": dalpha}

    return params, grads, mean_loss_history


def predict_log(X, alpha, K):
    y_log = logistic(K @ alpha.reshape(-1, 1))
    return y_log.squeeze()


def predict(X_test, alpha, K):
    y_log = predict_log(X_test, alpha, K)
    # Discretizing probabilities into 1s and 0s
    y_pred = y_log.round().reshape(1, -1)
    return y_pred
