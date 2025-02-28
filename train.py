import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load Data
df = pd.read_csv('data/churn_data.csv')  # Make sure you have a dataset in the 'data/' folder

# Convert categorical values to numeric
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Select features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save Model
pickle.dump(model, open("models/churn_model.pkl", "wb"))