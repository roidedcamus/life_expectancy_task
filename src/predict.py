import argparse
import pickle
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_df
from train_model import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--metrics_output_path', required=True)
    parser.add_argument('--predictions_output_path', required=True)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully: {type(model).__name__}")
    
    # Load and preprocess data    
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    X, y_true = preprocess_df(df, target='Life expectancy')
    print(f" Data loaded and preprocessed. Shape: {X.shape}")

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0).replace(0, 1)
    X = (X - mean) / std
    
    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    
    # Predict    
    print("Making predictions...")
    y_pred = model.predict(X.values)
    print(f" Predictions generated for {len(y_pred)} samples")
    
    # Save predictions (single column, no header)
    print(f"Saving predicions to: {args.predictions_output_path}")
    np.savetxt(args.predictions_output_path, y_pred, delimiter=',', fmt='%.6f')
    print("Predictions saved")
    
    # Calculate metrics
    print("Calculating metrics...")
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Save metrics
    print(f"Saving metrics to: {args.metrics_output_path}")
    with open(args.metrics_output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")
    print("Metrics saved")
    
    # Display metrics in terminal as well
    print("\n=== RESULTS ===")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²) Score: {r2:.2f}")
    print("=================")

if __name__ == "__main__":
    main()