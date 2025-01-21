import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global variables to store model and scaler
model = None
scaler = None
feature_columns = ['Temperature', 'Run_Time', 'Vibration']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic manufacturing data for demonstration"""
    np.random.seed(42)
    data = {
        'Machine_ID': np.random.randint(1, 11, n_samples),
        'Temperature': np.random.normal(75, 15, n_samples),
        'Run_Time': np.random.normal(100, 30, n_samples),
        'Vibration': np.random.normal(0.5, 0.2, n_samples),
    }
    
    # Generate Downtime_Flag based on conditions
    temp_factor = (data['Temperature'] > 90).astype(int)
    runtime_factor = (data['Run_Time'] > 150).astype(int)
    vibration_factor = (data['Vibration'] > 0.7).astype(int)
    
    # Combine factors with some randomness
    probability = (temp_factor + runtime_factor + vibration_factor) / 3 + np.random.normal(0, 0.1, n_samples)
    data['Downtime_Flag'] = (probability > 0.5).astype(int)
    
    return pd.DataFrame(data)

@app.route('/generate_sample', methods=['GET'])
def generate_sample():
    """Generate and save sample dataset"""
    df = generate_synthetic_data()
    sample_path = os.path.join(UPLOAD_FOLDER, 'sample_data.csv')
    df.to_csv(sample_path, index=False)
    return jsonify({'message': 'Sample dataset generated successfully', 'path': sample_path})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return '''
            <html>
                <body>
                    <form action="/upload" method="post" enctype="multipart/form-data">
                        <input type="file" name="file">
                        <input type="submit" value="Upload">
                    </form>
                </body>
            </html>
        '''
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename
            })
        
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/train', methods=['POST'])
def train_model():
    """Train model endpoint"""
    global model, scaler
    
    try:
        # Load data
        filename = request.json.get('filename', 'sample_data.csv')
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        data = pd.read_csv(filepath)
        
        # Prepare features and target
        X = data[feature_columns]
        y = data['Downtime_Flag']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Save model and scaler
        joblib.dump(model, os.path.join(MODEL_FOLDER, 'model.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'scaler.pkl'))
        
        return jsonify({
            'message': 'Model trained successfully',
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    global model, scaler
    
    try:
        # Load model and scaler if not already loaded
        if model is None:
            model = joblib.load(os.path.join(MODEL_FOLDER, 'model.pkl'))
            scaler = joblib.load(os.path.join(MODEL_FOLDER, 'scaler.pkl'))
        
        # Get input data
        data = request.json
        input_data = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Scale input data
        input_scaled = scaler.transform(input_data[feature_columns])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'Downtime': 'Yes' if prediction == 1 else 'No',
            'Confidence': float(probability)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)