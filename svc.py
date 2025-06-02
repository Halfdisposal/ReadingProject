import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, kernel='linear', C=1.0, max_iter=1000, tol=1e-3):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.bias = 0
        
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        else:
            raise ValueError("Kernel not recognized. Use 'linear'")
    
    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
                
        return K
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        
        # Compute the kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Define the objective function and its gradient
        def objective(alphas):
            return 0.5 * np.sum(np.outer(y, y) * K * np.outer(alphas, alphas)) - np.sum(alphas)
        
        def gradient(alphas):
            return np.dot(np.outer(y, y) * K, alphas) - 1
        
        # Constraints: 0 <= alpha_i <= C and sum(alpha_i * y_i) = 0
        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': lambda alphas: np.dot(alphas, y)}
        
        # Solve the quadratic programming problem
        result = minimize(
            objective,
            np.zeros(n_samples),
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        # Extract the Lagrange multipliers
        alphas = result.x
        
        # Identify support vectors (points with alpha > 0)
        sv_indices = np.where(alphas > 1e-5)[0]
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Compute the bias term
        sv_boundary_indices = np.where((self.alphas > 1e-5) & (self.alphas < self.C - 1e-5))[0]
        
        if len(sv_boundary_indices) > 0:
            # Calculate bias using points on the margin
            bias_values = []
            for i in sv_boundary_indices:
                sv = self.support_vectors[i]
                sv_y = self.support_vector_labels[i]
                
                # Calculate decision value without bias
                decision_value = 0
                for alpha, s_vector, s_label in zip(self.alphas, self.support_vectors, self.support_vector_labels):
                    decision_value += alpha * s_label * self._kernel_function(sv, s_vector)
                
                # The bias for this support vector
                bias_values.append(sv_y - decision_value)
            
            self.bias = np.mean(bias_values)
        else:
            # Fallback if no points are on the margin
            bias_values = []
            for i in range(len(self.support_vectors)):
                sv = self.support_vectors[i]
                sv_y = self.support_vector_labels[i]
                
                decision_value = 0
                for alpha, s_vector, s_label in zip(self.alphas, self.support_vectors, self.support_vector_labels):
                    decision_value += alpha * s_label * self._kernel_function(sv, s_vector)
                
                bias_values.append(sv_y - decision_value)
            
            self.bias = np.mean(bias_values)
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            decision_value = 0
            for alpha, sv, sv_y in zip(self.alphas, self.support_vectors, self.support_vector_labels):
                decision_value += alpha * sv_y * self._kernel_function(X[i], sv)
            predictions[i] = decision_value + self.bias
        
        return np.sign(predictions)

# Example usage
if __name__ == "__main__":
    # Generate random data for binary classification
    np.random.seed(42)
    X_class1 = np.random.randn(50, 2) - 2
    X_class2 = np.random.randn(50, 2) + 2
    X = np.vstack((X_class1, X_class2))
    y = np.hstack((np.ones(50) * -1, np.ones(50)))
    
    # Train the SVM
    svm = SVM(kernel='linear', C=1.0)
    svm.fit(X, y)
    
    # Make predictions
    y_pred = svm.predict(X)
    accuracy = np.mean(y_pred == y) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Number of support vectors: {len(svm.support_vectors)}")
