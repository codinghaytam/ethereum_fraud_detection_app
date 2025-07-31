
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class AddressFraudClassifier(nn.Module):
    """LSTM-based neural network for address-level fraud classification using transaction sequences"""
    
    def __init__(self, sequence_input_size, static_input_size=0, hidden_size=128, num_layers=2, 
                 fc_hidden_sizes=[256, 128], num_classes=2, dropout_rate=0.3):
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


def load_model(model_path):
    """Load the trained model - simplified version using only the .pth file"""
    
    # Initialize model with the same configuration as training script
    # Based on fraudClassifier.py configuration and actual saved model
    model = AddressFraudClassifier(
        sequence_input_size=10,  # From training script
        static_input_size=22,    # Actual number of static features from training (not 45)
        hidden_size=128,
        num_layers=4,            # From training script
        fc_hidden_sizes=[256, 128],
        num_classes=2,
        dropout_rate=0.3
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Create a simple StandardScaler for static features (will be fitted on-the-fly if needed)
    static_scaler = StandardScaler()
    
    # Define static feature columns based on what was actually used in training (22 features)
    # These are the first 22 features from the original list that were available in the dataset
    static_feature_cols = [
        'Avg min between sent tnx', 'Avg min between received tnx', 
        'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
        'Number of Created Contracts', 'Unique Received From Addresses', 
        'Unique Sent To Addresses', 'min value received', 'max value received ',
        'avg val received', 'min val sent', 'max val sent', 'avg val sent',
        'min value sent to contract', 'max val sent to contract', 
        'avg value sent to contract', 'total transactions (including tnx to create contract',
        'total Ether sent', 'total ether received', 'total ether sent contracts',
        'total ether balance'
    ]
    
    # Define sequence feature names (10 features based on training script)
    sequence_features = [
        'gas', 'gas_price', 'gas_used', 'value', 'is_error', 
        'txreceipt_status', 'hour_of_day', 'day_of_week', 'time_diff', 'created_at'
    ]
    
    # No sequence scaler used in this implementation
    sequence_scaler = None
    
    # Simple config
    config = {
        'sequence_input_size': 10,
        'static_input_size': len(static_feature_cols),  # Now correctly 22
        'hidden_size': 128,
        'num_layers': 4,
        'sequence_length': 50
    }
    
    print(f"Loaded model with {len(static_feature_cols)} static features and {len(sequence_features)} sequence features")
    
    return model, static_scaler, sequence_scaler, static_feature_cols, sequence_features, config


def prepare_etherscan_transactions(transactions, sequence_features, static_feature_cols, max_sequence_length=50):
    """
    Prepare Etherscan transaction data for the fraud detection model - simplified version
    
    Args:
        transactions: List of transaction dictionaries from Etherscan API
        sequence_features: List of sequence feature names from trained model
        static_feature_cols: List of static feature column names from trained model
        max_sequence_length: Maximum sequence length (default 50)
    
    Returns:
        sequence_array: Preprocessed sequence data ready for model input
        static_array: Preprocessed static features ready for model input
    """
    if not transactions:
        return None, None
    
    try:
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(transactions)
        print(f"Processing {len(df)} transactions...")
        
        # ========== SEQUENCE FEATURE PREPARATION ==========
        
        # Convert string fields to numeric
        numeric_fields = ['gas', 'gasPrice', 'gasUsed', 'value', 'isError', 'txreceipt_status']
        for field in numeric_fields:
            if field in df.columns:
                df[f'{field}_numeric'] = pd.to_numeric(df[field], errors='coerce').fillna(0)
            else:
                df[f'{field}_numeric'] = 0
        
        # Handle timestamp conversion (Etherscan provides Unix timestamp)
        if 'timeStamp' in df.columns:
            df['timestamp_parsed'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
            df = df.sort_values('timestamp_parsed')  # Sort by time for proper sequence
            
            # Create time-based features
            df['hour_of_day'] = df['timestamp_parsed'].dt.hour
            df['day_of_week'] = df['timestamp_parsed'].dt.dayofweek
            df['timestamp_numeric'] = df['timeStamp'].astype(int)
            df['time_diff'] = df['timestamp_numeric'].diff().fillna(0)
            df['created_at'] = (df['timestamp_parsed'] - pd.Timestamp('2015-01-01')).dt.days  # Days since Ethereum start
        else:
            print("Warning: No timestamp field found")
            df['hour_of_day'] = 0
            df['day_of_week'] = 0
            df['time_diff'] = 0
            df['created_at'] = 0
        
        # Build sequence features array - 10 features as expected by model
        sequence_data = []
        
        # Extract the 10 expected sequence features
        feature_mapping = {
            'gas': 'gas_numeric',
            'gas_price': 'gasPrice_numeric', 
            'gas_used': 'gasUsed_numeric',
            'value': 'value_numeric',
            'is_error': 'isError_numeric',
            'txreceipt_status': 'txreceipt_status_numeric',
            'hour_of_day': 'hour_of_day',
            'day_of_week': 'day_of_week',
            'time_diff': 'time_diff',
            'created_at': 'created_at'
        }
        
        for feature in sequence_features:
            if feature in feature_mapping and feature_mapping[feature] in df.columns:
                sequence_data.append(df[feature_mapping[feature]].values)
            else:
                # If feature not found, fill with zeros
                print(f"Warning: Sequence feature '{feature}' not found, filling with zeros")
                sequence_data.append(np.zeros(len(df)))
        
        if not sequence_data:
            print("Error: No sequence features could be extracted")
            return None, None
        
        # Create sequence array (transactions x features)
        sequence_array = np.array(sequence_data).T
        
        # Pad or truncate to max_sequence_length
        if len(sequence_array) > max_sequence_length:
            sequence_array = sequence_array[-max_sequence_length:]  # Take most recent
            print(f"Truncated to {max_sequence_length} most recent transactions")
        elif len(sequence_array) < max_sequence_length:
            # Pad with zeros at the beginning
            padding = np.zeros((max_sequence_length - len(sequence_array), len(sequence_features)))
            sequence_array = np.vstack([padding, sequence_array])
            print(f"Padded sequence from {len(df)} to {max_sequence_length} transactions")
        
        # ========== STATIC FEATURE PREPARATION (SIMPLIFIED) ==========
        
        # For Etherscan data, we'll calculate basic aggregated features
        # and fill missing ones with zeros to match the expected 22 features
        
        basic_stats = {
            'total_transactions': len(df),
            'avg_gas': df.get('gas_numeric', pd.Series([0])).mean(),
            'max_gas': df.get('gas_numeric', pd.Series([0])).max(),
            'avg_gas_price': df.get('gasPrice_numeric', pd.Series([0])).mean(),
            'max_gas_price': df.get('gasPrice_numeric', pd.Series([0])).max(),
            'avg_gas_used': df.get('gasUsed_numeric', pd.Series([0])).mean(),
            'total_gas_used': df.get('gasUsed_numeric', pd.Series([0])).sum(),
            'avg_value': df.get('value_numeric', pd.Series([0])).mean(),
            'max_value': df.get('value_numeric', pd.Series([0])).max(),
            'total_value': df.get('value_numeric', pd.Series([0])).sum(),
            'error_rate': df.get('isError_numeric', pd.Series([0])).mean(),
            'unique_to_addresses': df['to'].nunique() if 'to' in df.columns else 0,
            'time_span': df.get('timestamp_numeric', pd.Series([0])).max() - df.get('timestamp_numeric', pd.Series([0])).min(),
            'avg_time_between_tx': (df.get('timestamp_numeric', pd.Series([0])).max() - df.get('timestamp_numeric', pd.Series([0])).min()) / max(len(df) - 1, 1),
            'min_value': df.get('value_numeric', pd.Series([0])).min(),
            'max_time_diff': df.get('time_diff', pd.Series([0])).max(),
            'avg_hour': df.get('hour_of_day', pd.Series([0])).mean(),
            'unique_days': df.get('day_of_week', pd.Series([0])).nunique(),
            'tx_frequency': len(df) / max((df.get('timestamp_numeric', pd.Series([0])).max() - df.get('timestamp_numeric', pd.Series([0])).min()) / 86400, 1),  # transactions per day
            'gas_efficiency': df.get('gasUsed_numeric', pd.Series([0])).sum() / max(df.get('gas_numeric', pd.Series([1])).sum(), 1),  # gas used / gas limit ratio
            'value_concentration': df.get('value_numeric', pd.Series([0])).std() / max(df.get('value_numeric', pd.Series([0])).mean(), 1),  # coefficient of variation
            'total_ether_balance': df.get('value_numeric', pd.Series([0])).sum()  # simplified ether balance
        }
        
        # Create static features array - exactly 22 features to match the trained model
        static_features_array = []
        stat_names = list(basic_stats.keys())
        
        for i in range(22):  # Exactly 22 features
            if i < len(stat_names):
                value = basic_stats[stat_names[i]]
            else:
                # If we have fewer calculated stats than needed, fill with 0
                value = 0
            
            static_features_array.append(float(value) if not pd.isna(value) else 0.0)
        
        static_array = np.array(static_features_array).reshape(1, -1)  # Reshape for single sample
        
        print(f"Generated sequence shape: {sequence_array.shape}, static shape: {static_array.shape}")
        
        return sequence_array, static_array
        
    except Exception as e:
        print(f"Error in prepare_etherscan_transactions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def predict_fraud_probability(model, sequence_data, static_data, sequence_scaler, static_scaler, device='cpu'):
    """
    Make fraud prediction using preprocessed data - simplified version
    
    Args:
        model: Trained fraud detection model
        sequence_data: Preprocessed sequence array from prepare_etherscan_transactions
        static_data: Preprocessed static array from prepare_etherscan_transactions
        sequence_scaler: Fitted sequence scaler (can be None)
        static_scaler: Static scaler object (will be fitted on the fly if needed)
        device: PyTorch device ('cpu' or 'cuda')
    
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    model.to(device)
    
    try:
        with torch.no_grad():
            # Validate input data
            if sequence_data is None:
                raise ValueError("Sequence data is None")
            
            # Prepare sequence data (no scaling for now)
            sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)
            sequence_tensor = sequence_tensor.to(device)
            
            # Prepare static data
            static_tensor = None
            if static_data is not None and len(static_data[0]) > 0:
                # For static features, apply simple normalization instead of fitted scaler
                # This avoids the issue of having a pre-fitted scaler
                static_normalized = static_data.copy()
                
                # Simple min-max normalization to [0, 1] range
                for i in range(static_normalized.shape[1]):
                    col_data = static_normalized[:, i]
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max > col_min:
                        static_normalized[:, i] = (col_data - col_min) / (col_max - col_min)
                    else:
                        static_normalized[:, i] = 0.0
                
                static_tensor = torch.FloatTensor(static_normalized).to(device)
            
            # Make prediction
            outputs = model(sequence_tensor, static_tensor)
            
            # Check for NaN or invalid outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Warning: Invalid model outputs detected, using conservative prediction")
                return {
                    'is_fraud': False,
                    'fraud_probability': 0.5,
                    'non_fraud_probability': 0.5,
                    'confidence': 0.0,
                    'status': 'warning',
                    'message': 'Model output contained invalid values'
                }
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Validate probabilities
            if torch.isnan(probabilities).any():
                print("Warning: Invalid probabilities detected")
                return {
                    'is_fraud': False,
                    'fraud_probability': 0.5,
                    'non_fraud_probability': 0.5,
                    'confidence': 0.0,
                    'status': 'error',
                    'message': 'Invalid probability calculation'
                }
            
            fraud_prob = float(probabilities[0][1].item())
            non_fraud_prob = float(probabilities[0][0].item())
            confidence = float(torch.max(probabilities).item())
            
            return {
                'is_fraud': bool(predicted.item()),
                'fraud_probability': fraud_prob,
                'non_fraud_probability': non_fraud_prob,
                'confidence': confidence,
                'status': 'success',
                'prediction_details': {
                    'model_output': outputs.cpu().numpy().tolist()[0],
                    'sequence_shape': list(sequence_data.shape),
                    'static_features_count': len(static_data[0]) if static_data is not None else 0,
                    'used_sequence_scaler': sequence_scaler is not None,
                    'used_static_normalization': static_data is not None
                }
            }
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'is_fraud': False,
            'fraud_probability': 0.0,
            'non_fraud_probability': 1.0,
            'confidence': 0.0,
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }
