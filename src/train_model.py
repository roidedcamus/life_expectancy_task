import numpy as np
import pandas as pd
from data_preprocessing import preprocess_df

# --- MODEL IMPLEMENTATIONS --- 
class LinearRegression:
    """Linear Regression implementation using normal equation."""
    def __init__(self):
        self.coefficients = None  # weights (θ1, θ2, ...)
        self.intercept = None     # bias term (θ0)
    
    def fit(self, X, y):
        """
        Train the model using normal equation: θ = Inverse of(Transpose of X x X)(Transpose of X)(y)
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
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
    """Ridge Regression (L2 regularization) implementation from scratch"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # regularization strength
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """Train with L2 regularization: θ = Inverse of(Transpose of X x X + αI)(Transpose of X)(y)"""
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        I = np.eye(X_with_bias.shape[1])  # Identity matrix
        I[0, 0] = 0  # Don't regularize the intercept term
        
        # Ridge regression formula
        theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias) + self.alpha * I).dot(X_with_bias.T).dot(y)
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.coefficients) + self.intercept

# --- UTILITY FUNCTIONS ---
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and validation sets"""
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_size))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

def calculate_mse(y_true, y_pred):
    """Calculate MSE"""
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
    df = pd.read_csv('../../data/Life Expectancy.csv')
    X, y = preprocess_df(df)
    
    print("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    
    print("Training models...")
    # Experiment with different models
    models = {
        'linear_regression': LinearRegression(),
        'ridge_alpha_1.0': RidgeRegression(alpha=1.0),
        'ridge_alpha_0.1': RidgeRegression(alpha=0.1),
        'ridge_alpha_10.0': RidgeRegression(alpha=10.0)
    }
    
    best_mse = float('inf')
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train.values, y_train.values)  # Convert to numpy arrays
        y_pred = model.predict(X_val.values)
        mse = calculate_mse(y_val, y_pred)
        print(f"{name} Validation MSE: {mse:.2f}")
        
        # Track best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = name
    
    # Save the best model
    print(f"\nBest model: {best_model_name} with MSE: {best_mse:.2f}")
    save_model(best_model, '../models/regression_model_final.pkl')
    print("Model saved to ../models/regression_model_final.pkl")

if __name__ == "__main__":
    main()