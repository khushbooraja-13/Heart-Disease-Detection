from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('model/heart_disease_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    print(f"Error loading model/scaler: {str(e)}")
    raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data in correct order
        feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        features = [float(request.form[field]) for field in feature_order]
        final_features = [np.array(features)]
        
        # Scale features
        scaled_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Interpret result
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        
        return render_template('result.html', prediction_text=result)
    
    except Exception as e:
        return render_template('result.html', 
                             prediction_text=f"Error: {str(e)}. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)