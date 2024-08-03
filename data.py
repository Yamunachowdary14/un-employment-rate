import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("Book3.0.csv")

# Convert 'TIME' column to integer
data['TIME'] = data['TIME'].astype(int)

# Print unique years to debug
print("Unique years in the 'TIME' column:", data['TIME'].unique())

# Columns to remove
columns_to_remove = ['LOCATION', 'INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY']

# Clean up column names by removing whitespaces
data.columns = data.columns.str.strip()

# Drop unwanted columns
data = data.drop(columns=columns_to_remove, errors='ignore')

# Drop rows with null values
data = data.dropna()

# Split the data into features (X) and target variable (y)
X = data[['TIME']]
y = data['Value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the pickled model
with open('linear_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Determine the range of years in the dataset
min_year = data['TIME'].min()
max_year = data['TIME'].max()

# User input: provide the time for prediction
while True:
    input_time_str = input(f"Enter the prediction year ({min_year}-{max_year}), or type 'quit' to exit: ")
    if input_time_str.lower() == 'quit':
        break
    try:
        input_time = int(input_time_str)
        if input_time < min_year or input_time > max_year:
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter a valid year within the range.")
        continue

    # Predict the target variable for the provided input time
    predicted_value = loaded_model.predict([[input_time]])

    print(f"Predicted Unemployment Rate for {input_time}: {predicted_value[0]}")
