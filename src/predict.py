import pandas as pd
import joblib
import sys

def load_model(model_filepath):
    """Load the trained model from a file."""
    return joblib.load(model_filepath)

def make_prediction(model, input_data):
    """Make a prediction using the trained model."""
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Load model
    model_path = "../models/churn_model.pkl"
    model = load_model(model_path)
    
    # Sample input data (Replace with real user input or file input)
    input_data = pd.DataFrame([[34, 120, 2, 1, 0, 1, 50]],
                              columns=["Age", "MonthlyCharges", "TotalServices", "Gender", "Partner", "Dependents", "Tenure"])
    
    # Make prediction
    prediction = make_prediction(model, input_data)
    print(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
