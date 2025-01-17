from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        fbs = float(request.form['fbs'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        
        # Manually scale the input features using the loaded scaler
        data = np.array([[age, sex, fbs, trestbps, chol]])
        data_scaled = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(data_scaled)
        
        # Map prediction to meaningful result
        if prediction[0] == 1:
            result = 'Hypertension prone'
        else:
            result = 'Not Hypertension prone'
        
        return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
