import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def svm_regression_dual(X, t, C, epsilon=0.1):
    N = len(t)
    K = (X @ X.T)
    
    x0 = np.zeros(2 * N)
    
    def objective(x):
        a = x[:N]
        a_hat = x[N:]
        diff = a - a_hat
        term1 = -0.5 * diff @ (K @ diff)
        term2 = -epsilon * np.sum(a + a_hat)
        term3 = -t @ (a - a_hat)
        return -(term1 + term2 - term3)
    
    def grad(x):
        a = x[:N]
        a_hat = x[N:]
        diff = a - a_hat
        grad_a = K @ diff + epsilon - t
        grad_a_hat = -K @ diff + epsilon + t
        return np.concatenate([grad_a, grad_a_hat])
    
    constraints = [
        {
            'type': 'eq',
            'fun': lambda x: np.sum(x[:N] - x[N:]),
            'jac': lambda x: np.concatenate([np.ones(N), -np.ones(N)])
        }
    ]
    
    bounds = [(0, C)] * (2 * N)
    
    result = minimize(
        fun=objective,
        x0=x0,
        method='SLSQP',
        jac=grad,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': True, 'maxiter': 1000}
    )
    
    if not result.success:
        raise RuntimeError(f"{result.message}")
    
    a_opt = result.x[:N]
    a_hat_opt = result.x[N:]
    return a_opt, a_hat_opt

def calculate_weights(X, a, a_hat):
    return np.sum((a - a_hat).reshape(-1, 1) * X, axis=0)

def calculate_bias(X, t, w, a, a_hat, epsilon, C):
    sv_idx = np.where(((0 < a) & (a < C)) | ((0 < a_hat) & (a_hat < C)))[0]
    
    if len(sv_idx) == 0:
        sv_idx = np.arange(len(t))
    
    b_values = []
    for i in sv_idx:
        if 0 < a[i] < C:
            b_values.append(t[i] - np.dot(w, X[i]) - epsilon)
        elif 0 < a_hat[i] < C:
            b_values.append(t[i] - np.dot(w, X[i]) + epsilon)
    return np.mean(b_values) if b_values else 0

def predict(X, w, b):
    return X @ w + b

if __name__ == "__main__":
    N = 100
    X, t = make_regression(n_samples=N, n_features=1, noise=10)
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)
    
    C = 1000.0
    epsilon = 0.01
    
    a, a_hat = svm_regression_dual(X_train, t_train, C, epsilon)
    
    w = calculate_weights(X_train, a, a_hat)
    b = calculate_bias(X_train, t_train, w, a, a_hat, epsilon, C)
    
    print("Weight w:", w)
    print("Bias b:", b)
    
    X_range = X_train
    t_range = X_range @ w + b
    
    fig, ax = plt.subplots(figsize=(10, 6),)
    ax.set_facecolor("#D1D1FFFF")
    ax.scatter(X_train, t_train, color="#FF68C5", label='Train Data')
    ax.scatter(X_test, t_test, color="#5FFFB4", label='Test Data')
    ax.plot(X_range, t_range, color="#093EFF", label='SVM Fit')

    ax.set_title('SVM Regression')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.legend()
    ax.grid(True, alpha=0.3, color = "#FFFFFF")
    plt.show()
