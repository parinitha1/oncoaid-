"""
Data Preprocessing Module for Bone Cancer Classification

This module provides utilities for getting valid categorical values and preprocessing input data for the bone cancer classification application. Data loading and preprocessing are handled in main.py. This module does not interact with the Gemini LLM integration for cancer prevention tips, which is managed in app.py.
"""

from sklearn.preprocessing import LabelEncoder
import pickle
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_valid_values():
    """
    Get valid values for each categorical feature.

    Returns:
        dict: Dictionary mapping feature names to their valid values
    """
    try:
        logger.info("Fetching valid values for categorical features...")
        valid_values = {
            'Sex': ['Female', 'Male'],
            'Grade': ['High', 'Intermediate'],
            'Histological type': [
                'epithelioid sarcoma',
                'leiomyosarcoma',
                'malignant solitary fibrous tumor',
                'myxofibrosarcoma',
                'myxoid fibrosarcoma',
                'pleiomorphic leiomyosarcoma',
                'pleiomorphic spindle cell undifferentiated',
                'pleomorphic sarcoma',
                'poorly differentiated synovial sarcoma',
                'sclerosing epithelioid fibrosarcoma',
                'synovial sarcoma',
                'undifferentiated - pleiomorphic',
                'undifferentiated pleomorphic liposarcoma'
            ],
            'MSKCC type': ['Leiomyosarcoma', 'MFH', 'Synovial sarcoma'],
            'Site of primary STS': [
                'left biceps',
                'left buttock',
                'left thigh',
                'parascapusular',
                'right buttock',
                'right parascapusular',
                'right thigh'
            ],
            'Treatment': [
                'Radiotherapy + Surgery',
                'Surgery + Chemotherapy',
                'Radiotherapy + Surgery + Chemotherapy'
            ]
        }
        logger.info("✅ Valid values fetched successfully")
        return valid_values
    except Exception as e:
        logger.error(f"Error fetching valid values: {str(e)}")
        raise

def preprocess_input(input_df):
    """
    Preprocess input data for prediction.

    Args:
        input_df (pandas.DataFrame): Input DataFrame with features to preprocess

    Returns:
        numpy.ndarray: Preprocessed and scaled feature array
    """
    try:
        logger.info("Preprocessing input data...")
        
        # Validate required columns
        required_columns = ['Age', 'Sex', 'Grade', 'Histological type', 'MSKCC type', 'Site of primary STS']
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in input data: {', '.join(missing_columns)}")
        
        # Log input data
        logger.info(f"Input data columns: {list(input_df.columns)}")
        logger.info(f"Input data values: {input_df.iloc[0].to_dict()}")
        
        # Load preprocessing objects
        preprocessing_path = 'backend/preprocessing.pkl'
        if not os.path.exists(preprocessing_path):
            raise FileNotFoundError(
                f"Preprocessing objects file not found at {preprocessing_path}. "
                "Please run train_all_models.py to generate the preprocessing objects."
            )
        
        with open(preprocessing_path, 'rb') as f:
            preprocessing = pickle.load(f)
            scaler = preprocessing['scaler']
            label_encoders = preprocessing['label_encoders']
        
        # Validate input values
        possible_values = get_valid_values()
        for col in input_df.columns:
            if col in label_encoders:
                value = input_df[col].iloc[0]
                if value not in possible_values[col]:
                    raise ValueError(f"Invalid value for {col}. Valid values are: {', '.join(possible_values[col])}")
        
        # Encode categorical features
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Scale features
        X = input_df.values
        X = scaler.transform(X)
        
        logger.info("✅ Input data preprocessed successfully")
        return X
    
    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise