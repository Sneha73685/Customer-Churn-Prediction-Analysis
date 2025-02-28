import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data_filepath, model_filepath):
    # Load processed dataset
    df = pd.read_csv(data_filepath)
    
    # Define features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save trained model
    joblib.dump(model, model_filepath)
    print(f"Model saved successfully at: {model_filepath}")

if __name__ == "__main__":
    train_model("../data/processed_churn_data.csv", "../models/churn_model.pkl")