import warnings
import json
import datetime
import time
import requests
import pickle

# Import core libraries with error handling
try:
    import numpy as np
    print("✓ NumPy loaded successfully")
except ImportError as e:
    print(f"✗ Error importing NumPy: {e}")
    raise

try:
    import pandas as pd
    print("✓ Pandas loaded successfully") 
except ImportError as e:
    print(f"✗ Error importing Pandas: {e}")
    raise

try:
    import torch
    import torch.nn as nn
    print("✓ PyTorch loaded successfully")
except ImportError as e:
    print(f"✗ Error importing PyTorch: {e}")
    raise

try:
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import Dataset, DataLoader
    print("✓ ML libraries loaded successfully")
except ImportError as e:
    print(f"✗ Error importing ML libraries: {e}")
    raise

warnings.filterwarnings('ignore')

# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EtherscanAPI:
    """Simple wrapper for Etherscan API to fetch transaction data"""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.rate_limit_delay = 0.2  # 200ms delay between requests to respect rate limits

    def get_transactions(self, address, start_block=0, end_block=99999999, page=1, offset=10000):
       
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'  # Most recent first
        }

        if self.api_key:
            params['apikey'] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            time.sleep(self.rate_limit_delay)  # Rate limiting

            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    return data['result']
                else:
                    print(f"API Error: {data.get('message', 'Unknown error')}")
                    return []
            else:
                print(f"HTTP Error: {response.status_code}")
                return []

        except Exception as e:
            print(f"Error fetching transactions: {e}")
            return []

    def get_internal_transactions(self, address, start_block=0, end_block=99999999, page=1, offset=10000):
        """Fetch internal transactions for an address"""
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }

        if self.api_key:
            params['apikey'] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            time.sleep(self.rate_limit_delay)

            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    return data['result']
                else:
                    return []
            else:
                return []

        except Exception as e:
            print(f"Error fetching internal transactions: {e}")
            return []

    def get_erc20_transfers(self, address, start_block=0, end_block=99999999, page=1, offset=10000):
        """Fetch ERC20 token transfers for an address"""
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }

        if self.api_key:
            params['apikey'] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            time.sleep(self.rate_limit_delay)

            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    return data['result']
                else:
                    return []
            else:
                return []

        except Exception as e:
            print(f"Error fetching ERC20 transfers: {e}")
            return []

class TransactionAnalyzer:

    @staticmethod
    def extract_static_features(address, normal_txs):
        
        features = {}
        
        # Convert address to lowercase for comparison
        address = address.lower()
        
        # Normal transaction analysis
        sent_txs = [tx for tx in normal_txs if tx['from'].lower() == address]
        received_txs = [tx for tx in normal_txs if tx['to'].lower() == address]
        
        # Basic counts
        features['Sent tnx'] = len(sent_txs)
        features['Received Tnx'] = len(received_txs)
        features['Number of Created Contracts'] = len([tx for tx in sent_txs if tx['to'] == ''])
        
        # Unique addresses
        unique_sent_to = set(tx['to'].lower() for tx in sent_txs if tx['to'])
        unique_received_from = set(tx['from'].lower() for tx in received_txs)
        features['Unique Sent To Addresses'] = len(unique_sent_to)
        features['Unique Received From Addresses'] = len(unique_received_from)
        
        # Value analysis (convert from wei to ether)
        def safe_float_conversion(value):
            """Safely convert value to float, handling edge cases"""
            try:
                if not value or value == '':
                    return 0.0
                return float(value) / 1e18
            except (ValueError, TypeError):
                return 0.0
        
        sent_values = [safe_float_conversion(tx['value']) for tx in sent_txs]
        received_values = [safe_float_conversion(tx['value']) for tx in received_txs]
        
        features['min val sent'] = min(sent_values) if sent_values else 0
        features['max val sent'] = max(sent_values) if sent_values else 0
        features['avg val sent'] = sum(sent_values) / len(sent_values) if sent_values else 0
        features['total Ether sent'] = sum(sent_values)
        
        features['min value received'] = min(received_values) if received_values else 0
        features['max value received '] = max(received_values) if received_values else 0  # Note: space in name matches training data
        features['avg val received'] = sum(received_values) / len(received_values) if received_values else 0
        features['total ether received'] = sum(received_values)
        
        # Time analysis
        def safe_int_conversion(value):
            """Safely convert value to int, handling edge cases"""
            try:
                if not value or value == '':
                    return 0
                return int(value)
            except (ValueError, TypeError):
                return 0
        
        if normal_txs:
            timestamps = [safe_int_conversion(tx['timeStamp']) for tx in normal_txs]
            timestamps = [ts for ts in timestamps if ts > 0]  # Filter out invalid timestamps
            timestamps.sort()
            
            if len(timestamps) > 1:
                features['Time Diff between first and last (Mins)'] = (timestamps[-1] - timestamps[0]) / 60
                
                sent_timestamps = [safe_int_conversion(tx['timeStamp']) for tx in sent_txs]
                received_timestamps = [safe_int_conversion(tx['timeStamp']) for tx in received_txs]
                
                # Filter out invalid timestamps
                sent_timestamps = [ts for ts in sent_timestamps if ts > 0]
                received_timestamps = [ts for ts in received_timestamps if ts > 0]
                
                if len(sent_timestamps) > 1:
                    sent_diffs = [(sent_timestamps[i] - sent_timestamps[i-1]) / 60 for i in range(1, len(sent_timestamps))]
                    features['Avg min between sent tnx'] = sum(sent_diffs) / len(sent_diffs)
                else:
                    features['Avg min between sent tnx'] = 0
                    
                if len(received_timestamps) > 1:
                    received_diffs = [(received_timestamps[i] - received_timestamps[i-1]) / 60 for i in range(1, len(received_timestamps))]
                    features['Avg min between received tnx'] = sum(received_diffs) / len(received_diffs)
                else:
                    features['Avg min between received tnx'] = 0
            else:
                features['Time Diff between first and last (Mins)'] = 0
                features['Avg min between sent tnx'] = 0
                features['Avg min between received tnx'] = 0
        else:
            features['Time Diff between first and last (Mins)'] = 0
            features['Avg min between sent tnx'] = 0
            features['Avg min between received tnx'] = 0
        
        # Contract interaction analysis
        def safe_hex_to_int(hex_str):
            """Safely convert hex string to int, handling edge cases"""
            if not hex_str or hex_str == '0x':
                return 0
            try:
                return int(hex_str, 16)
            except ValueError:
                return 0
        
        contract_txs = [tx for tx in sent_txs if tx['to'] and safe_hex_to_int(tx['input']) != 0]
        contract_values = [safe_float_conversion(tx['value']) for tx in contract_txs]
        
        features['min value sent to contract'] = min(contract_values) if contract_values else 0
        features['max val sent to contract'] = max(contract_values) if contract_values else 0
        features['avg value sent to contract'] = sum(contract_values) / len(contract_values) if contract_values else 0
        features['total ether sent contracts'] = sum(contract_values)
        
        # Total transaction count
        features['total transactions (including tnx to create contract'] = len(normal_txs)
        
        # Balance calculation (approximation)
        features['total ether balance'] = features['total ether received'] - features['total Ether sent']
        
        return features

    @staticmethod
    def create_transaction_sequence(normal_txs, sequence_length=100):
        
        all_txs = []
        
        # Add normal transactions
        for tx in normal_txs:
            try:
                all_txs.append({
                    'timestamp': int(tx['timeStamp']) if tx['timeStamp'] else 0,
                    'value': float(tx['value']) / 1e18 if tx['value'] else 0.0,
                    'gas': int(tx['gas']) if tx['gas'] else 0,
                    'gasPrice': int(tx['gasPrice']) if tx['gasPrice'] else 0,
                    'gasUsed': int(tx['gasUsed']) if tx['gasUsed'] and tx['gasUsed'] != '' else 0,
                    'isError': int(tx['isError']) if tx['isError'] and tx['isError'] != '' else 0,
                    'txType': 0,  # Normal transaction
                    'blockNumber': int(tx['blockNumber']) if tx['blockNumber'] else 0,
                    'transactionIndex': int(tx['transactionIndex']) if tx['transactionIndex'] else 0,
                    'confirmations': int(tx['confirmations']) if tx['confirmations'] and tx['confirmations'] != '' else 0
                })
            except (ValueError, TypeError) as e:
                # Skip invalid transactions
                print(f"Warning: Skipping invalid transaction: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        all_txs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Take only the required number of transactions
        all_txs = all_txs[:sequence_length]
        
        # Convert to feature matrix
        features = []
        for tx in all_txs:
            # Create feature vector matching training data (10 features)
            feature_vector = [
                tx['timestamp'] % 1000000,  # Normalized timestamp
                tx['value'],
                tx['gas'] / 1e6,  # Normalized gas
                tx['gasPrice'] / 1e9,  # Normalized gas price (Gwei)
                tx['gasUsed'] / 1e6,  # Normalized gas used
                tx['isError'],
                tx['txType'],
                tx['blockNumber'] % 1000000,  # Normalized block number
                tx['transactionIndex'],
                tx['confirmations'] / 100  # Normalized confirmations
            ]
            features.append(feature_vector)
        
        # Pad with zeros if we have fewer transactions than sequence_length
        while len(features) < sequence_length:
            features.append([0.0] * 10)
        
        return np.array(features, dtype=np.float32)

class FraudClassifier(nn.Module):
    """Fraud Detection Model Architecture - matches your training script exactly"""

    def __init__(self, sequence_input_size, static_input_size=0, hidden_size=128, num_layers=2,
                 fc_hidden_sizes=[256, 128], num_classes=2, dropout_rate=0.3):
        super(FraudClassifier, self).__init__()

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

        # Combine with static features if available
        if static_features is not None and self.static_input_size > 0:
            combined_features = torch.cat([lstm_features, static_features], dim=1)
        else:
            combined_features = lstm_features

        # Pass through classifier
        output = self.classifier(combined_features)
        return output


class FraudDetectionModel:
 

    def __init__(self, model_dir='./model/', etherscan_api_key=None):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.config = None
        self.device = device
        
        # Initialize Etherscan API
        self.etherscan = EtherscanAPI(etherscan_api_key)
        self.analyzer = TransactionAnalyzer()

       
        self._load_model()

    def _load_model(self):
        """Load the trained model and all components"""
        try:
            print("Loading fraud detection model...")

            # Load model configuration
            with open(f'{self.model_dir}model_config.pkl', 'rb') as f:
                self.config = pickle.load(f)

            # Load static feature scaler
            with open(f'{self.model_dir}static_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # Load feature column names
            with open(f'{self.model_dir}static_feature_columns.pkl', 'rb') as f:
                saved_columns = pickle.load(f)

            # Use only the features the scaler expects
            expected_features = self.scaler.n_features_in_
            self.feature_columns = saved_columns[:expected_features]

            print(f"✓ Loaded {len(self.feature_columns)} static features")
            print(f"✓ Model expects {expected_features} features")

            # Initialize model
            self.model = FraudClassifier(
                sequence_input_size=self.config['sequence_input_size'],
                static_input_size=self.config['static_input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                fc_hidden_sizes=self.config['fc_hidden_sizes'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout_rate']
            )

            # Load trained weights
            model_state = torch.load(f'{self.model_dir}/fraud_classifier.pth', map_location='cpu')
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()

            print("✓ Model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def analyze_address_from_etherscan(self, address, max_transactions=100):
       
        print(f"Analyzing address: {address}")
        
        try:
            # Fetch different types of transactions
            print("  - Fetching normal transactions...")
            normal_txs = self.etherscan.get_transactions(address, offset=max_transactions)
            
            # Extract static features
            print("Extracting features...")
            static_features = self.analyzer.extract_static_features(address, normal_txs)
            
            # Create transaction sequence
            transaction_sequence = self.analyzer.create_transaction_sequence(
                normal_txs, self.config['sequence_length']
            )
            
            # Make prediction
            result = self._predict_with_sequence(static_features, transaction_sequence)
            
            # Add transaction summary
            result['total_transactions'] = len(normal_txs)
            result['timestamp'] = datetime.datetime.now().isoformat()
            
            return {
                "result":result,
                "transactionsUsed":normal_txs
            }
            
        except Exception as e:
            print(f"Error analyzing address: {e}")
            return None

    def _predict_with_sequence(self, static_features, transaction_sequence):
        
        try:
            # Prepare static features
            feature_dict = static_features.copy()
            
            # Add missing features with zeros
            for col in self.feature_columns:
                if col not in feature_dict:
                    feature_dict[col] = 0.0

            # Create DataFrame and select features in correct order
            df = pd.DataFrame([feature_dict])
            features = df[self.feature_columns].values

            # Scale features
            features_scaled = self.scaler.transform(features).astype(np.float32)

            # Prepare transaction sequence
            if transaction_sequence.shape[0] == 1:
                sequence_tensor = torch.FloatTensor(transaction_sequence).to(self.device)
            else:
                sequence_tensor = torch.FloatTensor(transaction_sequence.reshape(1, *transaction_sequence.shape)).to(self.device)
            
            static_tensor = torch.FloatTensor(features_scaled).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(sequence_tensor, static_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1)

                # Get results
                fraud_prob = probabilities[0][1].cpu().item()
                normal_prob = probabilities[0][0].cpu().item()
                prediction = predicted_class[0].cpu().item()

            return {
                'prediction': 'Fraud' if prediction == 1 else 'Normal',
                'fraud_probability': fraud_prob,
                'normal_probability': normal_prob,
                'confidence': max(fraud_prob, normal_prob),
                'is_fraud': prediction == 1,
                'features_used': len(self.feature_columns),
                'sequence_length': transaction_sequence.shape[0] if len(transaction_sequence.shape) > 1 else 1
            }

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def predict_address(self, address_features):
       
        try:
            # Prepare features
            feature_dict = address_features.copy()

            # Add missing features with zeros
            for col in self.feature_columns:
                if col not in feature_dict:
                    feature_dict[col] = 0.0

            # Create DataFrame and select features in correct order
            df = pd.DataFrame([feature_dict])
            features = df[self.feature_columns].values

            # Scale features
            features_scaled = self.scaler.transform(features).astype(np.float32)

            # Create dummy transaction sequence
            sequence_length = self.config['sequence_length']
            sequence_input_size = self.config['sequence_input_size']
            dummy_sequence = np.zeros((1, sequence_length, sequence_input_size), dtype=np.float32)

            # Convert to tensors
            sequence_tensor = torch.FloatTensor(dummy_sequence).to(self.device)
            static_tensor = torch.FloatTensor(features_scaled).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(sequence_tensor, static_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1)

                # Get results
                fraud_prob = probabilities[0][1].cpu().item()
                normal_prob = probabilities[0][0].cpu().item()
                prediction = predicted_class[0].cpu().item()

            return {
                'prediction': 'Fraud' if prediction == 1 else 'Normal',
                'fraud_probability': fraud_prob,
                'normal_probability': normal_prob,
                'confidence': max(fraud_prob, normal_prob),
                'is_fraud': prediction == 1
            }

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def batch_analyze_addresses(self, addresses_list, max_transactions=1000, save_results=True):
        
        results = []
        
        print(f"Analyzing {len(addresses_list)} addresses...")
        
        for i, address in enumerate(addresses_list):
            print(f"\nAnalyzing address {i+1}/{len(addresses_list)}: {address}")
            
            result = self.analyze_address_from_etherscan(address, max_transactions)
            
            if result:
                result['address'] = address
                result['batch_index'] = i
                results.append(result)
                
                # Show quick result
                status = "🚨 FRAUD" if result['is_fraud'] else "✅ NORMAL"
                confidence = result['confidence']
                print(f"   Result: {status} (confidence: {confidence:.3f})")
            else:
                print(f"   ❌ Analysis failed for {address}")
            
            # Rate limiting - pause between addresses
            if i < len(addresses_list) - 1:
                time.sleep(1)  # 1 second between addresses
        
        # Save results if requested
        if save_results and results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fraud_analysis_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n✓ Results saved to {filename}")
        
        return results

    def get_feature_importance(self):
        """Get the list of features used by the model"""
        return {
            'static_features': self.feature_columns,
            'num_static_features': len(self.feature_columns),
            'sequence_length': self.config['sequence_length'],
            'sequence_input_size': self.config['sequence_input_size']
        }

    def get_model_info(self):
        """Get model configuration information"""
        return {
            'model_type': 'LSTM + Static Features with Etherscan Integration',
            'device': str(self.device),
            'config': self.config,
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'etherscan_enabled': True
        }


def load_fraud_model_with_etherscan(etherscan_api_key=None, model_dir='../model/'):
    
    return FraudDetectionModel(model_dir, etherscan_api_key)


def predict_address_state(address,apikey, modelDir):
    model = load_fraud_model_with_etherscan(etherscan_api_key=apikey, model_dir=modelDir)
    return model.analyze_address_from_etherscan(address)