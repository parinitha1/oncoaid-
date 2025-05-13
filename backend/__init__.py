"""
Backend package for the Bone Cancer Classification web application.

This package contains modules for training and managing classical, deep learning, and quantum models used in the bone cancer classification application. It does not handle the Gemini LLM integration for cancer prevention tips, which is managed in app.py.
"""

from .main import DeepNeuralNetwork, train_deep_model, train_classical_model, load_and_preprocess_data
from .quantum_models import QuantumMLModel, QuantumNeuralNetwork

__all__ = [
    'DeepNeuralNetwork',
    'train_deep_model',
    'train_classical_model',
    'load_and_preprocess_data',
    'QuantumMLModel',
    'QuantumNeuralNetwork'
]