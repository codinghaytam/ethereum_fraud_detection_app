#!/usr/bin/env python3
"""
Ethereum Address Fraud Prediction using Etherscan API

This module provides a function to predict fraud for any Ethereum address
by fetching transaction data from Etherscan API and using the trained model.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# API Configuration
DEFAULT_ETHERSCAN_API_KEY = "WT5E1QCQAEGK556D626MCTTNA7W4GERA7T"  # Get from https://etherscan.io/apis
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
API_REQUEST_TIMEOUT = 30
API_RATE_LIMIT_DELAY = 0.2  # Etherscan allows 5 requests per second

# Transaction Fetching Configuration
DEFAULT_MAX_TRANSACTIONS = 50

# Model Configuration
DEFAULT_MODEL_PATH = "../model/address_fraud_classifier_lstm.pth"
MAX_SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 2
FC_HIDDEN_SIZES = [256, 128]
DROPOUT_RATE = 0.3

# Feature Configuration
FRAUD_THRESHOLD = 0.5  # Threshold for fraud classification

# Expected static feature columns (in order)
EXPECTED_STATIC_FEATURES = [
    'total_transactions', 'avg_gas', 'max_gas', 'total_gas',
    'avg_gas_price', 'max_gas_price', 'avg_gas_used', 'total_gas_used',
    'avg_transaction_value', 'max_transaction_value', 'total_transaction_value',
    'error_rate', 'total_errors', 'avg_tx_status', 'time_span',
    'avg_time_between_tx', 'unique_to_addresses', 'unique_functions', 'unique_methods'
]

# Etherscan API response columns to convert to numeric
ETHERSCAN_NUMERIC_COLUMNS = [
    'blockNumber', 'timeStamp', 'nonce', 'value', 'gas', 'gasPrice', 
    'gasUsed', 'isError', 'txreceipt_status', 'confirmations'
]

# Column mapping for Etherscan API response
ETHERSCAN_COLUMN_MAPPING = {
    'from': 'from_address',
    'to': 'to_address',
    'timeStamp': 'timestamp_numeric'
}

# Example usage configuration
TEST_ADDRESS = "0x742dA6cCB3B4cB1BC8F6Ce1A9C5b5b3e1234567890"  # Replace with actual address

# =============================================================================
# END CONFIGURATION VARIABLES
# =============================================================================

class AddressFraudClassifier(nn.Module):
    """LSTM-based neural network for address-level fraud classification using transaction sequences"""
    
    def __init__(self, sequence_input_size, static_input_size=0, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, 
                 fc_hidden_sizes=FC_HIDDEN_SIZES, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(AddressFraudClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.static_input_size = static_input_size
        
        # LSTM for sequential transaction data
        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Calculate the combined feature size
        lstm_output_size = hidden_size * 2  # bidirectional
        combined_size = lstm_output_size + static_input_size
        
        # Fully connected layers
        fc_layers = []
        prev_size = combined_size
        
        for fc_hidden_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(fc_hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = fc_hidden_size
        
        # Output layer
        fc_layers.append(nn.Linear(prev_size, num_classes))
        
        self.classifier = nn.Sequential(*fc_layers)
        
    def forward(self, sequences, static_features=None):
        batch_size = sequences.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(sequences.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(sequences.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(sequences, (h0, c0))
        
        # Use the last output from LSTM
        lstm_features = lstm_out[:, -1, :]
        
        # Combine LSTM features with static features if available
        if static_features is not None and self.static_input_size > 0:
            combined_features = torch.cat([lstm_features, static_features], dim=1)
        else:
            combined_features = lstm_features
        
        # Pass through classifier
        output = self.classifier(combined_features)
        return output

def fetch_etherscan_transactions(address, api_key, max_transactions=DEFAULT_MAX_TRANSACTIONS):
    """
    Fetch transaction data from Etherscan API for a given address
    
    Args:
        address (str): Ethereum address to analyze
        api_key (str): Etherscan API key
        max_transactions (int): Maximum number of transactions to fetch
    
    Returns:
        pd.DataFrame: Transaction data
    """
    print(f"Fetching transactions for address: {address}")
    
    base_url = "https://api.etherscan.io/api"
    
    all_transactions = []
    page = 1
    offset = DEFAULT_MAX_TRANSACTIONS
    
    while len(all_transactions) < max_transactions:
        # Calculate start block for pagination
        start_block = (page - 1) * offset + 1 if page > 1 else 0
        
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': 99999999,
            'page': page,
            'offset': offset,
            'sort': 'asc',
            'apikey': api_key
        }
        
        try:
            response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=API_REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != '1':
                if 'No transactions found' in data.get('message', ''):
                    print("No transactions found for this address")
                    break
                else:
                    print(f"API Error: {data.get('message', 'Unknown error')}")
                    break
            
            transactions = data['result']
            
            if not transactions:
                print("No more transactions to fetch")
                break
            
            all_transactions.extend(transactions)
            print(f"Fetched {len(transactions)} transactions (total: {len(all_transactions)})")
            
            # If we got fewer transactions than requested, we've reached the end
            if len(transactions) < offset:
                break
            
            page += 1
            
            # Rate limiting - Etherscan allows 5 requests per second
            time.sleep(API_RATE_LIMIT_DELAY)
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    if not all_transactions:
        print("No transactions found for this address")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_transactions)
    
    # Convert numeric columns
    for col in ETHERSCAN_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamp to datetime
    if 'timeStamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timeStamp'], unit='s')
    
    # Rename columns to match model expectations
    df = df.rename(columns=ETHERSCAN_COLUMN_MAPPING)
    
    # Add source_address column (same as from_address for consistency)
    if 'from_address' in df.columns:
        df['source_address'] = df['from_address']
    
    print(f"Successfully fetched {len(df)} transactions")
    return df

def prepare_address_features(transaction_df, target_address, max_sequence_length=MAX_SEQUENCE_LENGTH):
    """
    Prepare features for the target address from transaction data
    
    Args:
        transaction_df (pd.DataFrame): Transaction data
        target_address (str): Address to analyze
        max_sequence_length (int): Maximum sequence length for LSTM
    
    Returns:
        tuple: (static_features, sequences, sequence_feature_names)
    """
    if transaction_df is None or len(transaction_df) == 0:
        return None, None, None
    
    print("Preparing address features...")
    
    # Debug: Print column names and sample data
    print(f"Transaction DataFrame columns: {list(transaction_df.columns)}")
    print(f"Sample from_address values: {transaction_df['from_address'].head().tolist()}")
    print(f"Sample to_address values: {transaction_df['to_address'].head().tolist()}")
    
    # Normalize address case
    target_address = target_address.lower()
    print(f"Looking for target address: {target_address}")
    
    # Filter transactions for the target address (as sender OR receiver)
    # The address can be either the sender (from) or receiver (to)
    from_mask = transaction_df['from_address'].str.lower() == target_address
    to_mask = transaction_df['to_address'].str.lower() == target_address
    
    address_txs = transaction_df[from_mask | to_mask].copy()
    
    print(f"Transactions where address is sender: {from_mask.sum()}")
    print(f"Transactions where address is receiver: {to_mask.sum()}")
    
    if len(address_txs) == 0:
        print("No transactions found for the target address")
        return None, None, None
    
    print(f"Found {len(address_txs)} transactions for address {target_address}")
    
    # Sort by timestamp
    if 'timestamp' in address_txs.columns:
        address_txs = address_txs.sort_values('timestamp')
    elif 'timestamp_numeric' in address_txs.columns:
        address_txs = address_txs.sort_values('timestamp_numeric')
    
    # Prepare sequence features
    sequence_features = []
    
    # Gas-related features
    if 'gas' in address_txs.columns:
        address_txs['gas_numeric'] = pd.to_numeric(address_txs['gas'], errors='coerce').fillna(0)
        sequence_features.append('gas_numeric')
    
    if 'gasPrice' in address_txs.columns:
        address_txs['gas_price_numeric'] = pd.to_numeric(address_txs['gasPrice'], errors='coerce').fillna(0)
        sequence_features.append('gas_price_numeric')
    elif 'gasPrice' in address_txs.columns:
        address_txs['gas_price_numeric'] = pd.to_numeric(address_txs['gasPrice'], errors='coerce').fillna(0)
        sequence_features.append('gas_price_numeric')
    
    if 'gasUsed' in address_txs.columns:
        address_txs['gas_used_numeric'] = pd.to_numeric(address_txs['gasUsed'], errors='coerce').fillna(0)
        sequence_features.append('gas_used_numeric')
    
    # Value features
    if 'value' in address_txs.columns:
        address_txs['value_numeric'] = pd.to_numeric(address_txs['value'], errors='coerce').fillna(0)
        sequence_features.append('value_numeric')
    
    # Error and status features
    if 'isError' in address_txs.columns:
        address_txs['is_error_numeric'] = pd.to_numeric(address_txs['isError'], errors='coerce').fillna(0)
        sequence_features.append('is_error_numeric')
    
    if 'txreceipt_status' in address_txs.columns:
        address_txs['txreceipt_status_numeric'] = pd.to_numeric(address_txs['txreceipt_status'], errors='coerce').fillna(0)
        sequence_features.append('txreceipt_status_numeric')
    
    # Time-based features
    if 'timestamp' in address_txs.columns:
        address_txs['hour_of_day'] = address_txs['timestamp'].dt.hour
        address_txs['day_of_week'] = address_txs['timestamp'].dt.dayofweek
        sequence_features.extend(['hour_of_day', 'day_of_week'])
        
        # Time differences
        address_txs['timestamp_numeric'] = address_txs['timestamp'].astype('int64') // 10**9
        address_txs['time_diff'] = address_txs['timestamp_numeric'].diff().fillna(0)
        sequence_features.append('time_diff')
    elif 'timestamp_numeric' in address_txs.columns:
        address_txs['time_diff'] = address_txs['timestamp_numeric'].diff().fillna(0)
        sequence_features.append('time_diff')
    
    print(f"Sequence features: {sequence_features}")
    
    if not sequence_features:
        print("No valid sequence features found")
        return None, None, None
    
    # Create sequence data
    seq_data = address_txs[sequence_features].values
    
    # Pad or truncate to max_sequence_length
    if len(seq_data) > max_sequence_length:
        seq_data = seq_data[-max_sequence_length:]  # Take most recent transactions
    elif len(seq_data) < max_sequence_length:
        padding = np.zeros((max_sequence_length - len(seq_data), len(sequence_features)))
        seq_data = np.vstack([padding, seq_data])
    
    # Prepare static features
    static_features = {}
    
    # Transaction count
    static_features['total_transactions'] = len(address_txs)
    
    # Gas-related aggregates
    if 'gas' in address_txs.columns:
        static_features['avg_gas'] = address_txs['gas'].mean()
        static_features['max_gas'] = address_txs['gas'].max()
        static_features['total_gas'] = address_txs['gas'].sum()
    
    if 'gasPrice' in address_txs.columns:
        static_features['avg_gas_price'] = address_txs['gasPrice'].mean()
        static_features['max_gas_price'] = address_txs['gasPrice'].max()
    
    if 'gasUsed' in address_txs.columns:
        static_features['avg_gas_used'] = address_txs['gasUsed'].mean()
        static_features['total_gas_used'] = address_txs['gasUsed'].sum()
    
    # Value-related aggregates
    if 'value' in address_txs.columns:
        value_numeric = pd.to_numeric(address_txs['value'], errors='coerce').fillna(0)
        static_features['avg_transaction_value'] = value_numeric.mean()
        static_features['max_transaction_value'] = value_numeric.max()
        static_features['total_transaction_value'] = value_numeric.sum()
    
    # Error and status aggregates
    if 'isError' in address_txs.columns:
        static_features['error_rate'] = address_txs['isError'].mean()
        static_features['total_errors'] = address_txs['isError'].sum()
    
    if 'txreceipt_status' in address_txs.columns:
        status_numeric = pd.to_numeric(address_txs['txreceipt_status'], errors='coerce').fillna(0)
        static_features['avg_tx_status'] = status_numeric.mean()
    
    # Time-based aggregates
    if 'timestamp_numeric' in address_txs.columns:
        static_features['time_span'] = address_txs['timestamp_numeric'].max() - address_txs['timestamp_numeric'].min()
        static_features['avg_time_between_tx'] = static_features['time_span'] / static_features['total_transactions'] if static_features['total_transactions'] > 0 else 0
    else:
        static_features['time_span'] = 0
        static_features['avg_time_between_tx'] = 0
    
    # Unique interactions
    if 'to_address' in address_txs.columns:
        static_features['unique_to_addresses'] = address_txs['to_address'].nunique()
    else:
        static_features['unique_to_addresses'] = 0
    
    # Function and method diversity (often not available in basic Etherscan data)
    static_features['unique_functions'] = 0
    static_features['unique_methods'] = 0
    
    # Fill any NaN values
    for key, value in static_features.items():
        if pd.isna(value):
            static_features[key] = 0
    
    static_features_array = np.array(list(static_features.values())).reshape(1, -1)
    sequences_array = seq_data.reshape(1, max_sequence_length, len(sequence_features))
    
    print(f"Static features shape: {static_features_array.shape}")
    print(f"Sequences shape: {sequences_array.shape}")
    
    return static_features_array, sequences_array, sequence_features

def predict_address_fraud(ethereum_address, etherscan_api_key, model_path=DEFAULT_MODEL_PATH):
    """
    Predict fraud probability for an Ethereum address using transaction history
    
    Args:
        ethereum_address (str): Ethereum address to analyze (with or without 0x prefix)
        etherscan_api_key (str): Etherscan API key
        model_path (str): Path to the trained model file
    
    Returns:
        dict: {
            'address': str,
            'prediction': str ('Fraud' or 'Not Fraud'),
            'confidence': float (0-1),
            'fraud_probability': float (0-1),
            'total_transactions': int,
            'error': str or None
        }
    """
    
    try:
        # Normalize address format
        if not ethereum_address.startswith('0x'):
            ethereum_address = '0x' + ethereum_address
        ethereum_address = ethereum_address.lower()
        
        # Load the trained model
        print("Loading trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            return {
                'address': ethereum_address,
                'prediction': None,
                'confidence': None,
                'fraud_probability': None,
                'total_transactions': 0,
                'error': f"Failed to load model: {str(e)}"
            }
        
        # Extract model configuration
        config = checkpoint['model_config']
        static_scaler = checkpoint.get('static_scaler')
        sequence_scaler = checkpoint.get('sequence_scaler')
        static_feature_cols = checkpoint.get('static_feature_cols', [])
        
        # Initialize model
        model = AddressFraudClassifier(
            sequence_input_size=config['sequence_input_size'],
            static_input_size=config['static_input_size'],
            hidden_size=config.get('hidden_size', HIDDEN_SIZE),
            num_layers=config.get('num_layers', NUM_LAYERS),
            num_classes=config.get('num_classes', NUM_CLASSES)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        
        # Fetch transaction data from Etherscan
        transaction_df = fetch_etherscan_transactions(ethereum_address, etherscan_api_key)
        
        if transaction_df is None or len(transaction_df) == 0:
            return {
                'address': ethereum_address,
                'prediction': "Fraud",
                'confidence': 1.0,
                'fraud_probability': 1.0,
                'total_transactions': 0,
                'error': "No transactions found for this address"
            }
        
        total_transactions = len(transaction_df)
        
        # Prepare features
        static_features, sequences, _ = prepare_address_features(
            transaction_df, ethereum_address, max_sequence_length=MAX_SEQUENCE_LENGTH
        )
        
        
        # Preprocess features using saved scalers
        if static_scaler is not None and len(static_feature_cols) > 0:
            # Create DataFrame with expected features, filling missing ones with zeros
            static_features_df = pd.DataFrame(static_features, columns=EXPECTED_STATIC_FEATURES)
            
            # Align features with training features
            aligned_features = pd.DataFrame(index=[0])
            for feature in static_feature_cols:
                if feature in static_features_df.columns:
                    aligned_features[feature] = static_features_df[feature].iloc[0]
                else:
                    aligned_features[feature] = 0
            
            static_features_scaled = static_scaler.transform(aligned_features)
        else:
            static_features_scaled = static_features
        
        # Scale sequences
        if sequence_scaler is not None:
            seq_reshaped = sequences.reshape(-1, sequences.shape[-1])
            seq_scaled = sequence_scaler.transform(seq_reshaped)
            sequences_scaled = seq_scaled.reshape(sequences.shape)
        else:
            sequences_scaled = sequences
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(sequences_scaled).to(device)
        static_features_tensor = torch.FloatTensor(static_features_scaled).to(device) if static_features_scaled is not None else None
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            outputs = model(sequences_tensor, static_features_tensor)
            
            # Handle NaN values
            if torch.isnan(outputs).any():
                outputs = torch.nan_to_num(outputs, nan=0.0)
            
            probabilities = torch.softmax(outputs, dim=1)
            
            if torch.isnan(probabilities).any():
                probabilities = torch.nan_to_num(probabilities, nan=0.5)
            
            fraud_probability = probabilities[0, 1].item()  # Probability of fraud (class 1)
            confidence = max(probabilities[0]).item()  # Confidence (max probability)
            prediction = "Fraud" if fraud_probability > FRAUD_THRESHOLD else "Not Fraud"
        
        prediction_result = {
            'address': ethereum_address,
            'prediction': prediction,
            'confidence': confidence,
            'fraud_probability': fraud_probability,
            'total_transactions': total_transactions,
            'transaction_used': transaction_df.to_json(),
            'error': None
        }

        
        return prediction_result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            'address': ethereum_address,
            'prediction': None,
            'confidence': None,
            'fraud_probability': None,
            'total_transactions': 0,
            'error': f"Prediction failed: {str(e)}"
        }

