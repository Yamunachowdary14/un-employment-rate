from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pickled model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Extract the year from the form
        year = int(request.form['year'])

        # Use the model to make a prediction (Replace this with your actual model prediction logic)
        prediction = model.predict([[year]])  # Update this line based on your model input format

        # Render the home page with the prediction
        return render_template('home.html', prediction=prediction[0])
    
@app.route('/statistics', methods=['GET'])
def statistics():
    # Render the statistics page
    return render_template('statistics.html')
if __name__ == '__main__':
    app.run(debug=True)
