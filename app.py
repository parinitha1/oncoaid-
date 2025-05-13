"""
Bone Cancer Classification Web Application
This module serves as the main entry point for the Flask web application.
It handles model loading, request processing, prediction generation, and personalized cancer prevention tips using the Gemini LLM.
"""

from flask import Flask, request, jsonify, render_template, redirect, session
import pandas as pd
import numpy as np
import pickle
import torch
import os
import logging
from backend.main import DeepNeuralNetwork, train_deep_model, load_and_preprocess_data
from backend.quantum_models import QuantumMLModel, QuantumNeuralNetwork
from werkzeug.security import generate_password_hash, check_password_hash
import pennylane as qml
import sqlite3
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_secret_key_here'

# Configure Gemini LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBdaJswEUdECyc7c_6DaDbx-2Z-2KwfaI8")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable not set")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini LLM configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini LLM: {str(e)}")
    # Fallback to default tips if Gemini configuration fails
    gemini_model = None

# Global variables for models and preprocessing objects
classical_model = None
dl_model = None
qml_model = None
qnn_model = None
scaler = None
label_encoders = None

# Default prevention tips (fallback if LLM fails)
DEFAULT_PREVENTION_TIPS = [
    "Maintain a balanced diet rich in calcium and vitamin D to support bone health.",
    "Engage in regular weight-bearing exercises like walking or jogging to strengthen bones.",
    "Avoid smoking, as it can weaken bones and increase cancer risk.",
    "Limit alcohol consumption to reduce overall cancer risk.",
    "Schedule regular check-ups to monitor bone health, especially with a family history of cancer."
]

# Create the database and users table
def init_db():
    db_path = 'users.db'
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            email TEXT NOT NULL UNIQUE,
                            password TEXT NOT NULL
                        )''')
        conn.commit()
        conn.close()

# ======================
# 🔐 Signup
# ======================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([name, email, password]):
            logger.error("Missing required fields in signup form")
            return render_template('signup.html', error='All fields are required.')

        password_hash = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password_hash))
            conn.commit()
            conn.close()
            logger.info(f"User signed up successfully: {email}")
            return redirect('/login')
        except sqlite3.IntegrityError:
            logger.error(f"Signup failed: Email already exists - {email}")
            return render_template('signup.html', error='Email already exists.')
        except Exception as e:
            logger.error(f"Signup failed: {str(e)}")
            return render_template('signup.html', error='An error occurred. Please try again.')
    
    return render_template('signup.html')

# ======================
# 🔓 Login
# ======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([email, password]):
            logger.error("Missing required fields in login form")
            return render_template('login.html', error='All fields are required.')

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT password, name FROM users WHERE email = ?", (email,))
            result = cursor.fetchone()
            conn.close()

            if result and check_password_hash(result[0], password):
                session['email'] = email
                session['name'] = result[1]
                logger.info(f"User logged in successfully: {email}")
                return redirect('/')
            else:
                logger.error(f"Login failed: Invalid credentials for {email}")
                return render_template('login.html', error='Invalid credentials.')
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return render_template('login.html', error='An error occurred. Please try again.')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    email = session.get('email', 'Unknown')
    session.pop('email', None)
    session.pop('name', None)
    logger.info(f"User logged out: {email}")
    return redirect('/login')

# ======================
# 📦 Load Models
# ======================
def load_models():
    global classical_model, dl_model, qml_model, qnn_model, scaler, label_encoders
    
    try:
        with open('backend/preprocessing.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
            scaler = preprocessing['scaler']
            label_encoders = preprocessing['label_encoders']
        logger.info("✅ Preprocessing objects loaded successfully!")
        
        with open('backend/classical_model.pkl', 'rb') as f:
            classical_model = pickle.load(f)
        logger.info("✅ Classical ML model loaded successfully")
        
        n_classes = len(label_encoders['Treatment'].classes_)
        dl_model = DeepNeuralNetwork(input_size=6, hidden_size=64, output_size=n_classes)
        dl_model.load_state_dict(torch.load('backend/dl_model.pth'))
        dl_model.eval()
        logger.info("✅ Deep Learning model loaded successfully")
        
        qnn_model = QuantumNeuralNetwork(
            input_size=6,
            hidden_size=64,
            output_size=n_classes,
            n_qubits=8
        )
        qnn_model.load_state_dict(torch.load('backend/quantum_nn_model.pth'))
        qnn_model.eval()
        logger.info("✅ Quantum Neural Network model loaded successfully")
        
        qml_model = QuantumMLModel(n_qubits=8, n_layers=4, n_classes=n_classes)
        qml_model.load_state_dict(torch.load('backend/qml_model.pth'))
        qml_model.eval()
        logger.info("✅ Quantum ML model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.warning("Some models failed to load. The application may not function properly.")
        return False

if not load_models():
    logger.warning("Some models failed to load. The application may not function properly.")

@app.route('/')
def home():
    if 'email' in session:
        return render_template('index.html', name=session['name'])
    else:
        return redirect('/login')
@app.route('/about')
def about():
    """
    Render the About page with information about the web application.
    """
    return render_template('about.html')

# ======================
# 📞 Contact Page
# ======================
@app.route('/contact')
def contact():
    """
    Render the Contact page with a contact form or contact information.
    """
    return render_template('contact.html')


# ======================
# 🩺 Cancer Prevention Tips
# ======================
@app.route('/prevention', methods=['POST'])
def get_prevention_tips():
    try:
        logger.info("Received prevention tips request")
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'error': 'No data provided'}), 400
        
        logger.debug(f"Request data: {data}")
        
        required_fields = ['Age', 'Sex']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            logger.error(f"Missing required fields for prevention tips: {missing_fields}")
            return jsonify({'error': f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        prompt = (
            f"Provide a list of general tips to prevent bone cancer for a {data['Sex']} aged {data['Age']}."
            " Return the tips as a bulleted list."
        )
        logger.info(f"Generated LLM prompt: {prompt}")
        
        prevention_tips = DEFAULT_PREVENTION_TIPS  # Default fallback
        
        if gemini_model:
            try:
                response = gemini_model.generate_content(prompt)
                logger.info("Received response from Gemini LLM")
                logger.debug(f"Raw LLM response: {response.text}")
                
                prevention_tips = response.text.split('\n')
                prevention_tips = [tip.strip('- ').strip() for tip in prevention_tips if tip.strip()]
                
                if not prevention_tips:
                    logger.warning("LLM returned empty tips, using default tips")
                    prevention_tips = DEFAULT_PREVENTION_TIPS
            except Exception as llm_error:
                logger.error(f"LLM request failed: {str(llm_error)}")
                logger.info("Falling back to default prevention tips")
                prevention_tips = DEFAULT_PREVENTION_TIPS
        else:
            logger.warning("Gemini LLM not available, using default tips")
        
        logger.info(f"Returning {len(prevention_tips)} prevention tips")
        return jsonify({
            'prevention_tips': prevention_tips
        })
    
    except Exception as e:
        logger.error(f"Failed to fetch prevention tips: {str(e)}")
        return jsonify({'error': f'Failed to fetch prevention tips: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['Age', 'Sex', 'Grade', 'HistologicalType', 'MSKCCType', 'SiteOfPrimarySTS']
        for field in required_fields:
            if field not in data or not data[field]:
                logger.error(f"Missing or empty value for {field}")
                return jsonify({'error': f'Missing or empty value for {field}'}), 400
        
        input_df = pd.DataFrame([{
            'Age': float(data['Age']),
            'Sex': data['Sex'],
            'Grade': data['Grade'],
            'Histological type': data['HistologicalType'],
            'MSKCC type': data['MSKCCType'],
            'Site of primary STS': data['SiteOfPrimarySTS']
        }])
        
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    possible_values = list(label_encoders[col].classes_)
                    if input_df[col].iloc[0] not in possible_values:
                        logger.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
                        return jsonify({
                            'error': f'Invalid value for {col}. Valid values are: {", ".join(possible_values)}'
                        }), 400
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    logger.error(f"ValueError during label encoding for {col}: {str(e)}")
                    return jsonify({'error': f'Invalid value for {col}. Please select a valid option.'}), 400
        
        input_array = input_df.values
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_scaled = scaler.transform(input_array)
        
        input_tensor = torch.FloatTensor(input_scaled).to(torch.float32)
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        classical_pred = classical_model.predict(input_scaled)[0]
        classical_proba = classical_model.predict_proba(input_scaled)[0]
        
        with torch.no_grad():
            dl_output = dl_model(input_tensor)
            dl_pred = torch.argmax(dl_output, dim=1).item()
            dl_proba = torch.softmax(dl_output, dim=1).numpy()[0]
        
        with torch.no_grad():
            qml_input = torch.zeros((input_tensor.shape[0], 8), dtype=torch.float32)
            qml_input[:, :6] = input_tensor[:, :6]
            qml_output = qml_model(qml_input)
            qml_pred = torch.argmax(qml_output, dim=1).item()
            qml_proba = torch.softmax(qml_output, dim=1).numpy()[0]
        
        with torch.no_grad():
            qnn_output = qnn_model(input_tensor)
            qnn_pred = torch.argmax(qnn_output, dim=1).item()
            qnn_proba = torch.softmax(qnn_output, dim=1).numpy()[0]
        
        treatment_encoder = label_encoders['Treatment']
        classical_label = treatment_encoder.inverse_transform([classical_pred])[0]
        dl_label = treatment_encoder.inverse_transform([dl_pred])[0]
        qml_label = treatment_encoder.inverse_transform([qml_pred])[0]
        qnn_label = treatment_encoder.inverse_transform([qnn_pred])[0]
        
        all_treatments = list(treatment_encoder.classes_)
        
        response = {
            'input_data': {
                'Age': data['Age'],
                'Sex': data['Sex'],
                'Grade': data['Grade'],
                'HistologicalType': data['HistologicalType'],
                'MSKCCType': data['MSKCCType'],
                'SiteOfPrimarySTS': data['SiteOfPrimarySTS']
            },
            'recommendations': {
                'classical_ml': classical_label,
                'deep_learning': dl_label,
                'quantum_ml': qml_label,
                'quantum_nn': qnn_label
            },
            'all_treatments': all_treatments,
            'treatment_probabilities': {
                'classical_ml': classical_proba.tolist(),
                'deep_learning': dl_proba.tolist(),
                'quantum_ml': qml_proba.tolist(),
                'quantum_nn': qnn_proba.tolist()
            }
        }
        
        logger.info("Prediction completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)