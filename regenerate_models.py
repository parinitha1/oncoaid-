"""
Script to delete and regenerate all models for bone cancer classification.

This script handles the regeneration of classical, deep learning, and quantum models used in the bone cancer classification web application. It deletes existing model files and retrains them using the provided dataset. Note that this script does not interact with the Gemini LLM integration for cancer prevention tips.
"""

import os
import shutil
import torch
import logging
import numpy as np
from backend.main import train_deep_model, train_classical_model, load_and_preprocess_data
from backend.quantum_models import QuantumMLModel, QuantumNeuralNetwork
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def delete_existing_models():
    """Delete all existing model files."""
    model_files = [
        'backend/classical_model.pkl',
        'backend/dl_model.pth',
        'backend/quantum_nn_model.pth',
        'backend/qml_model.pth',
        'backend/preprocessing.pkl'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Deleted {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")

def train_and_save_models():
    """
    Train and save all models.

    Trains classical, deep learning, and quantum models, ensuring the backend directory exists before saving.
    """
    try:
        # Ensure backend directory exists
        os.makedirs('backend', exist_ok=True)
        logger.info("Ensured backend directory exists")

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_preprocess_data()
        
        # Train classical model
        logger.info("Training classical model...")
        classical_model = train_classical_model(X_train, y_train)
        logger.info("✅ Classical model trained and saved")
        
        # Train deep learning model
        logger.info("Training deep learning model...")
        dl_model = train_deep_model(X_train, y_train, input_size=6, hidden_size=64, output_size=3)
        logger.info("✅ Deep learning model trained and saved")
        
        # Train quantum neural network
        logger.info("Training quantum neural network...")
        n_classes = len(label_encoders['Treatment'].classes_)
        qnn_model = QuantumNeuralNetwork(
            input_size=6,
            hidden_size=64,
            output_size=n_classes,
            n_qubits=8
        )
        optimizer = optim.Adam(qnn_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(torch.float32)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Training loop for QNN
        qnn_model.train()
        num_epochs = 10  # Reduced for faster regeneration; adjust as needed
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = qnn_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            logger.info(f"QNN Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        torch.save(qnn_model.state_dict(), 'backend/quantum_nn_model.pth')
        logger.info("✅ Quantum neural network trained and saved")
        
        # Train quantum ML model
        logger.info("Training quantum ML model...")
        qml_model = QuantumMLModel(n_qubits=8, n_layers=4, n_classes=n_classes)
        optimizer = optim.Adam(qml_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Pad input to match qubit count
        X_train_padded = np.zeros((X_train.shape[0], 8))
        X_train_padded[:, :6] = X_train
        X_train_tensor = torch.FloatTensor(X_train_padded).to(torch.float32)
        
        # Training loop for QML
        qml_model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = qml_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            logger.info(f"QML Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        torch.save(qml_model.state_dict(), 'backend/qml_model.pth')
        logger.info("✅ Quantum ML model trained and saved")
        
        logger.info("✅ All models trained and saved successfully!")
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

if __name__ == "__main__":
    # Delete existing models
    delete_existing_models()
    
    # Train and save new models
    train_and_save_models()
    
    logger.info("✅ Model regeneration complete!")