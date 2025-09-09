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
        'ridge_alpha_10.0': RidgeRegression(alpha=10.0)
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
    
    # Save ALL models as required by the assignment
    save_model(models['linear_regression'], '../models/regression_model1.pkl')
    save_model(models['ridge_alpha_1.0'], '../models/regression_model2.pkl')
    save_model(models['ridge_alpha_0.1'], '../models/regression_model3.pkl')
    save_model(best_model, '../models/regression_model_final.pkl')

    print(f"\nBest model: {best_model_name} with MSE: {best_mse:.2f}")
    print("All models saved to models/ directory:")
    print("- regression_model1.pkl (Linear Regression)")
    print("- regression_model2.pkl (Ridge alpha=1.0)") 
    print("- regression_model3.pkl (Ridge alpha=0.1)")
    print("- regression_model_final.pkl (Best model: Ridge alpha=10.0)")

if __name__ == "__main__":
    main()