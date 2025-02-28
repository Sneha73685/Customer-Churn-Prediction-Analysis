import pickle
import pandas as pd

# Load Trained Model
model = pickle.load(open("models/churn_model.pkl", "rb"))

# Example Data (Modify this based on your dataset)
new_data = pd.DataFrame({
    'Tenure': [5],
    'MonthlyCharges': [75],
    'TotalCharges': [375],
    'Contract': [1]  # Encode contract type manually
})

# Predict
prediction = model.predict(new_data)
print("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
