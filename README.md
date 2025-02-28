# Customer Churn Prediction Analysis

## 📌 Project Overview
This project aims to predict customer churn using machine learning. It analyzes customer data and determines the likelihood of a customer leaving a service.

## 📁 Folder Structure
```
Customer-Churn-Prediction-Analysis/
│── data/                # Contains raw & processed data
│── models/              # Stores trained machine learning models
│── notebooks/           # Jupyter Notebooks for EDA & model development
│── src/                 # Source code for preprocessing, training, and prediction
│── app/                 # Flask-based API for deployment
│── README.md            # Project documentation
│── requirements.txt     # Python dependencies
│── .gitignore           # Files to ignore in Git
```

## 🔧 Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sneha73685/Customer-Churn-Prediction-Analysis.git
cd Customer-Churn-Prediction-Analysis
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage
### 1️⃣ Data Preprocessing
Run the preprocessing script to clean and prepare data:
```bash
python src/preprocess.py
```

### 2️⃣ Train the Model
```bash
python src/train_model.py
```

### 3️⃣ Make Predictions
```bash
python src/predict.py
```

### 4️⃣ Run the Web App (Flask API)
```bash
cd app
python main.py
```
Access the app at `http://127.0.0.1:5000/`

## 📊 Technologies Used
- Python (Pandas, Scikit-learn, Flask)
- Machine Learning (Random Forest Classifier)
- Deployment using Flask API

## 💡 Future Enhancements
- Implement a Streamlit UI for better user experience
- Hyperparameter tuning for improved accuracy
- Add more features for better predictions

## 🤝 Contributing
Feel free to fork and contribute! Open a pull request for review.

## 🏆 Acknowledgments
Special thanks to the open-source community for inspiring this project!
