import pandas as pd
import joblib

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_data(df, filepath):
    """Save DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Data saved successfully at: {filepath}")

def load_model(model_filepath):
    """Load the trained model from a file."""
    return joblib.load(model_filepath)

def save_model(model, model_filepath):
    """Save trained model to a file."""
    joblib.dump(model, model_filepath)
    print(f"Model saved successfully at: {model_filepath}")