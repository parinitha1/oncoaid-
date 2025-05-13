"""
Main Model Implementation Module

This module contains the implementation of classical and deep learning models for the bone cancer classification web application. Quantum models are handled in quantum_models.py. This module does not interact with the Gemini LLM integration for cancer prevention tips, which is managed in app.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
Path("backend").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Check if models exist, if not train them
def ensure_models_exist():
    """
    Ensure classical and deep learning models exist, training them if necessary.
    Quantum models are handled in quantum_models.py and trained in train_all_models.py.
    """
    model_files = [
        "backend/classical_model.pkl",
        "backend/dl_model.pth"
    ]
    
    if not all(os.path.exists(f) for f in model_files):
        logger.info("Models not found. Training new models...")
        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_preprocess_data()
            logger.info(f"Data loaded successfully. Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Save preprocessing objects
            with open("backend/preprocessing.pkl", "wb") as f:
                pickle.dump({
                    'scaler': scaler,
                    'label_encoders': label_encoders
                }, f)
            logger.info("✅ Preprocessing objects saved successfully")
            
            # Train classical ML model
            logger.info("Training classical ML model...")
            try:
                classical_model = train_classical_model(X_train, y_train)
                with open("backend/classical_model.pkl", "wb") as f:
                    pickle.dump(classical_model, f)
                logger.info("Classical model saved successfully!")
            except Exception as e:
                logger.error(f"Error training classical model: {str(e)}")
                raise
            
            # Train deep learning model
            logger.info("Training deep learning model...")
            try:
                n_classes = len(label_encoders['Treatment'].classes_)
                dl_model = train_deep_model(X_train, y_train, input_size=X_train.shape[1], hidden_size=64, output_size=n_classes)
                torch.save(dl_model.state_dict(), "backend/dl_model.pth")
                logger.info("Deep learning model saved successfully!")
            except Exception as e:
                logger.error(f"Error training deep learning model: {str(e)}")
                raise
            
            logger.info("✅ All models trained and saved successfully!")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    else:
        logger.info("✅ All models found!")

class DeepNeuralNetwork(nn.Module):
    """
    Deep Neural Network for Bone Cancer Classification.
    Implements a multi-layer neural network with batch normalization.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Deep Neural Network.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layers
            output_size (int): Number of output classes
        """
        super(DeepNeuralNetwork, self).__init__()
        
        # Define network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        
        # Regularization
        self.dropout = nn.Dropout(0.4)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Network output
        """
        # Ensure input has the correct shape for batch normalization
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Fourth layer
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Fifth layer
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc6(x)
        return x

# Load and preprocess dataset
def load_and_preprocess_data():
    """
    Load and preprocess the dataset with proper error handling.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders)
    """
    try:
        logger.info("🔄 Loading dataset...")
        dataset_path = "dataset/Bone Tumor Dataset.csv"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
            
        df = pd.read_csv(dataset_path)

        # Validate required columns
        required_columns = ['Age', 'Sex', 'Grade', 'Histological type', 'MSKCC type', 
                          'Site of primary STS', 'Treatment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Encode categorical features
        label_encoders = {}
        categorical_columns = ['Sex', 'Grade', 'Histological type', 'MSKCC type', 
                             'Site of primary STS', 'Treatment']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            logger.info(f"Encoded {col} values: {list(le.classes_)}")
        
        logger.info("✅ Categorical encoding complete.")
        
        # Split features and target
        X = df[['Age', 'Sex', 'Grade', 'Histological type', 'MSKCC type', 'Site of primary STS']]
        y = df['Treatment']
        
        # Handle class imbalance using SMOTE
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_resampled
        )
        
        # Scale features
        scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        logger.info("✅ Data preprocessing complete.")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def train_classical_model(X_train, y_train):
    """
    Train a classical machine learning model (XGBoost).
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        
    Returns:
        XGBClassifier: Trained model
    """
    # Initialize and train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    # Save model
    with open('backend/classical_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def train_deep_model(X_train, y_train, input_size, hidden_size, output_size):
    """
    Train the deep neural network.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        input_size (int): Number of input features
        hidden_size (int): Size of hidden layers
        output_size (int): Number of output classes
        
    Returns:
        DeepNeuralNetwork: Trained model
    """
    # Convert data to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    # Initialize model
    model = DeepNeuralNetwork(input_size, hidden_size, output_size)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    # Save model
    torch.save(model.state_dict(), "backend/dl_model.pth")
    
    return model

def plot_results(y_true, y_pred_classical, y_pred_dl):
    """
    Plot confusion matrices for classical and deep learning models.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred_classical (numpy.ndarray): Predictions from classical model
        y_pred_dl (numpy.ndarray): Predictions from deep learning model
    """
    plt.figure(figsize=(10, 5))
    
    # Plot confusion matrices
    plt.subplot(121)
    sns.heatmap(confusion_matrix(y_true, y_pred_classical), annot=True, fmt='d')
    plt.title('Classical ML Confusion Matrix')
    
    plt.subplot(122)
    sns.heatmap(confusion_matrix(y_true, y_pred_dl), annot=True, fmt='d')
    plt.title('Deep Learning Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('static/results_comparison.png')
    plt.close()

if __name__ == "__main__":
    ensure_models_exist()
    print("✅ All models have been trained and saved successfully!")