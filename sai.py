from flask import Flask, render_template, request
import pickle
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Check if the model file exists before loading
model_file_path = 'linear_regression_model.pkl'
if os.path.exists(model_file_path):
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
else:
    # Load data
    data = pd.read_csv("Book3.0.csv")

    # Columns to remove
    columns_to_remove = ['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes']

    # Clean up column names by removing whitespaces
    data.columns = data.columns.str.strip()

    # Drop unwanted columns
    data = data.drop(columns=columns_to_remove, errors='ignore')

    # Drop rows with null values
    data = data.dropna()

    # Convert 'TIME' to datetime
    data['TIME'] = pd.to_datetime(data['TIME'])

    # Select only the necessary features and target variable
    data = data[['TIME', 'Value']]

    # Split the data into features (X) and target variable (y)
    X = data[['TIME']]
    y = data['Value']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to a file using pickle
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    # You can add your login logic here
    return render_template('home.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        feature_time = pd.to_datetime(request.form['feature_time'])
    except ValueError:
        return render_template('error.html', message="Invalid date format")

    # Make a prediction using the loaded model
    prediction = model.predict([[feature_time]])

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
