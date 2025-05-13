Bone Cancer Classification
A web application for classifying bone cancer cases using multiple machine learning approaches, including classical, deep learning, and quantum models.
Features

Modern, responsive web interface
Real-time predictions from four different models:
Classical ML (XGBoost)
Deep Neural Network
Quantum ML Model
Quantum Neural Network


Easy-to-use form with dropdown menus for categorical features
Detailed probability distributions for each prediction
Treatment recommendations based on model predictions
Cancer prevention tips powered by Gemini LLM, providing general precautions to reduce the risk of developing bone cancer

Requirements

Python 3.9 or higher
Dependencies:Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
torch==2.0.1
pennylane==0.32.0
xgboost==1.7.6
matplotlib==3.7.2
google-cloud-aiplatform==1.62.0  # Added for Gemini LLM integration



Setup

Clone the repository:
git clone "To Be Inserted"
cd bone-cancer-classification


Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Set up Google Cloud for Gemini LLM:

Create a Google Cloud project and enable the Vertex AI API.
Set up authentication:
For local development, run:gcloud auth application-default login


For production, set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account key JSON file:export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"




Set your Google Cloud project ID:export GOOGLE_CLOUD_PROJECT="your-project-id"




Train the models:
python train_quantum.py


Run the Flask application:
python app.py


Open your browser and navigate to http://localhost:5000


Input Parameters
The application requires the following input parameters:

Age (numeric)
Sex (categorical):
Female
Male


Grade (categorical):
High
Intermediate


Histological Type (categorical):
epithelioid sarcoma
leiomyosarcoma
malignant solitary fibrous tumor
myxofibrosarcoma
myxoid fibrosarcoma
pleiomorphic leiomyosarcoma
pleiomorphic spindle cell undifferentiated
pleomorphic sarcoma
poorly differentiated synovial sarcoma
sclerosing epithelioid fibrosarcoma
synovial sarcoma
undifferentiated - pleiomorphic
undifferentiated pleomorphic liposarcoma


MSKCC Type (categorical):
Leiomyosarcoma
MFH
Synovial sarcoma


Site of Primary STS (categorical):
left biceps
left buttock
left thigh
parascapusular
right buttock
right parascapusular
right thigh



Output
The application provides predictions from all four models, including:

Predicted treatment type
Probability distribution for each possible treatment
Model-specific confidence scores
Cancer prevention tips accessible via the interface, fetched from Gemini LLM

Model Architecture
Classical ML Model

XGBoost classifier
100 estimators
Maximum depth of 10
Learning rate of 0.1

Deep Neural Network

Input layer: 6 neurons (one per feature)
Hidden layers: 3 layers of 64 neurons each
Output layer: 3 neurons (one per treatment type)
Dropout rate: 0.4
Batch normalization
ReLU activation

Quantum ML Model

8 qubits
4 variational layers
Enhanced entanglement patterns
Rotation gates with 4 parameters
Pauli-Z measurements

Quantum Neural Network

Classical layers:
Input: 6 neurons
Hidden: 64 neurons
Output: 3 neurons


Quantum circuit:
8 qubits
4 quantum layers
Enhanced entanglement
Rotation gates
Pauli-Z measurements



Technologies Used

Python 3.9+
Flask 2.3.3
XGBoost 1.7.6
Scikit-learn 1.3.0
PyTorch 2.0.1
PennyLane 0.32.0
Google Cloud Vertex AI (for Gemini LLM)
TailwindCSS
JavaScript

License
This project is licensed under the MIT License - see the LICENSE file for details.
