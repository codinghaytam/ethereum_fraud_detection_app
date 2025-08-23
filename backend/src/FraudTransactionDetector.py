import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from sklearn.preprocessing import Normalizer

class SimpleFraudDetector:
    """Simple Ethereum Fraud Detector"""
    
    def __init__(self,transactions):
        self.model = joblib.load("../model/gradient_boosting_model_random_tree.pkl")
        self.normalizer = Normalizer()
        self.transactions= transactions
    
    
    def make_features_for_transaction(self, tx_data, all_transactions, main_address):
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
        
        # Get all transactions for the address
        transactions = self.transactions
        if not transactions:
            return []
        
        
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
                
                # Store transaction info with confidence and hash
                transaction_predictions.append({
                    'address': other_address,
                    'confidence': float(fraud_confidence),
                    'transaction_hash': tx.get('hash', ''),
                    'transaction_type': tx_type,
                    'addresses_involved': [tx['from'], tx['to']] if 'from' in tx and 'to' in tx else []
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
        # Use a reasonable threshold (0.6) instead of median
        high_confidence_fraud = [tx for tx in fraudulent_transactions if tx['confidence'] > 0.6]

        
        # Calculate overall confidence and is_fraud flag
        if high_confidence_fraud:
            # Average confidence of fraudulent transactions
            confidence = sum(tx['confidence'] for tx in high_confidence_fraud) / len(high_confidence_fraud)
            is_fraud = True
            addresses_involved = [tx['addresses_involved'] for tx in high_confidence_fraud]
            addresses_involved = [addr for sublist in addresses_involved for addr in sublist]  # Flatten list
            addresses_involved = list(set(addresses_involved))  # Ensure unique addresses 
            
        else:
            confidence = 0.0
            is_fraud = False
            addresses_involved=set()
        
        return {
            'fraudulent_transactions': high_confidence_fraud,
            'confidence': confidence,
            'is_fraud': is_fraud,
            'addresses_involved': addresses_involved
        }


# Simple usage
def detect_fraud(address, api_key, transactions):
    """Simple function to detect fraud"""
    detector = SimpleFraudDetector(transactions)
    return detector.predict(address, api_key)

