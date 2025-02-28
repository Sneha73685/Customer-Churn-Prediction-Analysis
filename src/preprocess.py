import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def preprocess_data(input_filepath, output_filepath):
    # Load dataset
    df = pd.read_csv(input_filepath)
    
    # Encode categorical variables
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convert categorical features into numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Save the processed dataset
    df.to_csv(output_filepath, index=False)
    print(f"Data preprocessing complete! Processed file saved at: {output_filepath}")

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

def predict(input_data, model_filepath):
    # Load trained model
    model = joblib.load(model_filepath)
    
    # Make prediction
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    preprocess_data("../data/churn_data.csv", "../data/processed_churn_data.csv")
    train_model("../data/processed_churn_data.csv", "../models/churn_model.pkl")