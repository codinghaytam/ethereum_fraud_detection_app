import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from sklearn.preprocessing import Normalizer

class SimpleFraudDetector:
    """Simple Ethereum Fraud Detector"""
    
    def __init__(self):
        self.model = joblib.load("gradient_boosting_model_random_tree.pkl")
        self.normalizer = Normalizer()  # Create normalizer directly
    
    def get_transactions(self, address, api_key):
        """Get transactions from Etherscan"""
        url = "https://api.etherscan.io/api"
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'offset': 1000,  # Limit to 100 transactions
            'page': 1,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == '1':
            return data['result']
        return []
    
    def make_features_for_transaction(self, tx_data, all_transactions, main_address):
        """Convert single transaction to features"""
        # Create features for individual transaction
        features = []
        
        # Convert transaction data
        tx_value = float(tx_data['value']) / 1e18  # Wei to ETH
        tx_timestamp = pd.to_datetime(int(tx_data['timeStamp']), unit='s')
        
        # Basic features
        features.append(int(tx_data.get('confirmations', 0)))  # confirmations
        
        # Time features
        features.append(tx_timestamp.month)  # Month
        features.append(tx_timestamp.day)    # Day  
        features.append(tx_timestamp.hour)   # Hour
        
        # Transaction value features (use this transaction's values)
        features.append(tx_value)            # mean_value_received (this transaction)
        features.append(0)                   # variance_value_received (single tx)
        features.append(tx_value)            # total_received (this transaction)
        features.append(0)                   # time_diff_first_last_received (single tx)
        
        # Transaction count features (simplified for single transaction)
        features.append(1)                   # total_tx_sent (this transaction)
        features.append(0)                   # total_tx_sent_malicious
        features.append(1)                   # total_tx_sent_unique (this transaction)
        features.append(0)                   # total_tx_sent_malicious_unique
        features.append(0)                   # total_tx_received_malicious_unique
        
        return features
    
    def predict(self, address, api_key, count=10):
        """Get transactions and return ranked fraudulent addresses by confidence"""
        print(f"Getting transactions for address: {address}")
        
        # Get all transactions for the address
        transactions = self.get_transactions(address, api_key)
        if not transactions:
            return []
        
        print(f"Found {len(transactions)} transactions")
        
        # Analyze each transaction and collect fraud predictions
        transaction_predictions = []
        
        for i, tx in enumerate(transactions):
            try:
                # Determine the other party in the transaction
                if tx['from'].lower() == address.lower():
                    other_address = tx['to']
                    tx_type = 'outgoing'
                else:
                    other_address = tx['from'] 
                    tx_type = 'incoming'
                
                # Skip if other address is empty or invalid
                if not other_address or other_address == '0x':
                    continue
                
                # Make features for this specific transaction
                features = self.make_features_for_transaction(tx, transactions, address)
                
                # Normalize features
                features_normalized = self.normalizer.transform([features])
                
                # Get prediction probabilities
                proba = self.model.predict_proba(features_normalized)[0]
                fraud_confidence = proba[1] if len(proba) > 1 else 0  # Probability of fraud class
                
                # Store transaction info with confidence (simplified)
                transaction_predictions.append({
                    'address': other_address,
                    'confidence': float(fraud_confidence)
                })
                
            except Exception as e:
                print(f"Error processing transaction {i}: {str(e)}")
                continue
        
        # Sort by fraud confidence (highest first) and return top results
        fraudulent_transactions = sorted(
            transaction_predictions, 
            key=lambda x: x['confidence'], 
            reverse=True
        )[:count]
        
        # Filter to only include transactions with confidence > 0.5
        high_confidence_fraud = [tx for tx in fraudulent_transactions if tx['confidence'] > 0.8]
        
        return high_confidence_fraud


# Simple usage
def detect_fraud(address, api_key):
    """Simple function to detect fraud"""
    detector = SimpleFraudDetector()
    return detector.predict(address, api_key)

print(detect_fraud("0xe61df1f5b8dd4e4e2a874157c2c97daf7314b795", "CCBAMAAFB7SAJS42VSW79IJP1FF7ZVGG4R"))