import numpy as np
import pandas as pd
from data_preprocessing import preprocess_df

# --- MODEL IMPLEMENTATIONS --- 
class LinearRegression:
    """Linear Regression implementation from scratch using normal equation."""
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """Train the model using normal equation: θ = (XᵀX)⁻¹Xᵀy"""
        # Add bias term (column of 1s) to X
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Calculate coefficients using normal equation
        theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        # Separate intercept and coefficients
        self.intercept = theta[0]
        self.coefficients = theta[1:]
    
    def predict(self, X):
        """Make predictions: y_pred = Xθ + intercept"""
        return X.dot(self.coefficients) + self.intercept

class RidgeRegression:
    """Ridge Regression (L2 regularization) implementation from scratch."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """Train with L2 regularization: θ = (XᵀX + αI)⁻¹Xᵀy"""
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        I = np.eye(X_with_bias.shape[1])
        I[0, 0] = 0  # Don't regularize the intercept term
        
        theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias) + self.alpha * I).dot(X_with_bias.T).dot(y)
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.coefficients) + self.intercept

class LassoRegression:
    """Lasso Regression (L1 regularization) via a minimal ISTA solver."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        # center features and target; later recover intercept
        mu = X.mean(axis=0)
        Xc = X - mu
        y_mean = y.mean()
        yc = y - y_mean

        # step size: 1 / L, where L = largest eigenvalue of Xc^T Xc = (sigma_max)^2
        smax = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)[0]
        L = (smax ** 2) if smax > 0 else 1.0

        w = np.zeros(d)
        for _ in range(500):  # fixed small number of iterations
            grad = Xc.T @ (Xc @ w - yc)
            z = w - grad / L
            # soft-threshold
            w = np.sign(z) * np.maximum(np.abs(z) - self.alpha / L, 0.0)

        self.coefficients = w
        self.intercept = y_mean - mu @ w

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coefficients + self.intercept

# --- UTILITY FUNCTIONS ---
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and validation sets."""
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_size))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def save_model(model, filepath):
    """Save trained model using pickle."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

# --- MAIN TRAINING FUNCTION ---
def main():
    """Main training pipeline."""
    print("Loading and preprocessing data...")
    df = pd.read_csv('../data/train_data.csv')
    X, y = preprocess_df(df, target='Life expectancy')
    
    # Convert boolean columns to integers (0, 1)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    
    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Drop any rows with missing values
    valid_indices = y.notna() & ~X.isna().any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    print("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    
    print("Training models...")
    models = {
        'linear_regression': LinearRegression(),
        'ridge_alpha_1.0': RidgeRegression(alpha=1.0),
        'ridge_alpha_0.1': RidgeRegression(alpha=0.1),
        'ridge_alpha_10.0': RidgeRegression(alpha=10.0),
        'lasso_alpha_10' : LassoRegression(alpha=10.0),
    }
    
    best_mse = float('inf')
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_val.values)
        mse = calculate_mse(y_val, y_pred)
        print(f"{name} Validation MSE: {mse:.2f}")
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = name

    # After training all models, sort them by performance

    # --- Evaluate and Save Models ---
    print("\nSorting models by performance...")
    model_performance = []
    for name, model in models.items():
        y_pred = model.predict(X_val.values)
        mse = calculate_mse(y_val, y_pred)
        model_performance.append((name, model, mse))
    
    # Sort by MSE (ascending - lower is better)
    model_performance.sort(key=lambda x: x[2])
    
    # Save best model separately as "final"
    best_name, best_model, best_mse = model_performance[0]
    save_model(best_model, '../models/regression_model_final.pkl')
    print(f"- regression_model_final.pkl (Best: {best_name}, MSE: {best_mse:.2f})")
    
    # Save ALL models (except best) in order: regression_model1.pkl, regression_model2.pkl, ...
    for i, (name, model, mse) in enumerate(model_performance[1:], 1):
        save_model(model, f'../models/regression_model{i}.pkl')
        print(f"- regression_model{i}.pkl ({name}, MSE: {mse:.2f})")

if __name__ == "__main__":
    main()
    print("Models saved and Training complete.")