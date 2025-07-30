#!/usr/bin/env python3
"""
Fraud Explanation Data Transformer

This script loads JSON data containing Ethereum addresses with comments and transforms it 
into a Mistral fine-tuning dataset format. Each training example consists of:
- Input: Address + top 100 transaction features
- Output: Comment explaining why the address is flagged as fraudulent

The script filters out entries with empty comments and saves the result in JSON format.
"""

import json
import os
import requests
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class FraudExplanationDataTransformer:
    """Transforms fraud address data into Mistral fine-tuning format."""
    
    def __init__(self, 
                 addresses_json_path: str = "data/addresses-darklist.json",
                 etherscan_api_key: str = "",
                 output_path: str = "data/mistral_training_dataset.jsonl"):
        """
        Initialize the transformer with file paths and API key.
        
        Args:
            addresses_json_path: Path to JSON file with addresses and comments
            etherscan_api_key: Etherscan API key for fetching transaction data
            output_path: Path where the transformed dataset will be saved
        """
        self.addresses_json_path = addresses_json_path
        self.etherscan_api_key = etherscan_api_key
        self.output_path = output_path
        self.addresses_data = None
        self.etherscan_base_url = "https://api.etherscan.io/api"
        
    def load_data(self) -> None:
        """Load JSON dataset with addresses and comments."""
        try:
            with open(self.addresses_json_path, 'r', encoding='utf-8') as f:
                self.addresses_data = json.load(f)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise
            
    def get_transactions_from_etherscan(self, address: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch normal transactions for an address from Etherscan API.
        
        Args:
            address: Ethereum address to get transactions for
            limit: Maximum number of transactions to fetch (default 100)
            
        Returns:
            List of transaction dictionaries or None if error
        """
        if not self.etherscan_api_key:
            return None
            
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': limit,
            'sort': 'desc',
            'apikey': self.etherscan_api_key
        }
        
        try:
            response = requests.get(self.etherscan_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == '1' and 'result' in data:
                return data['result']
            else:
                return None
                
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            return None
    
    def get_transaction_features(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get transaction features for a specific address from Etherscan API.
        
        Args:
            address: Ethereum address to get features for
            
        Returns:
            Dictionary of transaction features or None if address not found
        """
        transactions = self.get_transactions_from_etherscan(address)
        
        if not transactions:
            return None
            
        # Calculate basic transaction statistics
        features = {}
        
        # Basic counts
        features['total_transactions'] = len(transactions)
        
        # Transaction values (in Wei, convert to Ether)
        values = []
        gas_prices = []
        gas_used = []
        
        sent_count = 0
        received_count = 0
        unique_to_addresses = set()
        unique_from_addresses = set()
        
        for tx in transactions:
            try:
                value_wei = int(tx.get('value', 0))
                value_eth = value_wei / 1e18
                values.append(value_eth)
                
                gas_price = int(tx.get('gasPrice', 0))
                gas_prices.append(gas_price)
                
                gas_used_tx = int(tx.get('gasUsed', 0))
                gas_used.append(gas_used_tx)
                
                from_addr = tx.get('from', '').lower()
                to_addr = tx.get('to', '').lower()
                current_addr = address.lower()
                
                if from_addr == current_addr:
                    sent_count += 1
                    unique_to_addresses.add(to_addr)
                
                if to_addr == current_addr:
                    received_count += 1
                    unique_from_addresses.add(from_addr)
                    
            except (ValueError, TypeError):
                continue
        
        # Statistical features
        if values:
            features['min_value'] = min(values)
            features['max_value'] = max(values)
            features['avg_value'] = sum(values) / len(values)
            features['total_value'] = sum(values)
        else:
            features['min_value'] = 0
            features['max_value'] = 0
            features['avg_value'] = 0
            features['total_value'] = 0
            
        if gas_prices:
            features['avg_gas_price'] = sum(gas_prices) / len(gas_prices)
            features['max_gas_price'] = max(gas_prices)
        else:
            features['avg_gas_price'] = 0
            features['max_gas_price'] = 0
            
        if gas_used:
            features['avg_gas_used'] = sum(gas_used) / len(gas_used)
            features['total_gas_used'] = sum(gas_used)
        else:
            features['avg_gas_used'] = 0
            features['total_gas_used'] = 0
        
        features['sent_transactions'] = sent_count
        features['received_transactions'] = received_count
        features['unique_to_addresses'] = len(unique_to_addresses)
        features['unique_from_addresses'] = len(unique_from_addresses)
        
        # Time-based features
        if len(transactions) > 1:
            timestamps = []
            for tx in transactions:
                try:
                    timestamp = int(tx.get('timeStamp', 0))
                    timestamps.append(timestamp)
                except (ValueError, TypeError):
                    continue
            
            if len(timestamps) > 1:
                timestamps.sort()
                time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                if time_diffs:
                    features['avg_time_between_tx'] = sum(time_diffs) / len(time_diffs) / 60  # in minutes
                    features['total_time_span'] = (timestamps[-1] - timestamps[0]) / 60  # in minutes
                else:
                    features['avg_time_between_tx'] = 0
                    features['total_time_span'] = 0
            else:
                features['avg_time_between_tx'] = 0
                features['total_time_span'] = 0
        else:
            features['avg_time_between_tx'] = 0
            features['total_time_span'] = 0
        
        # Add rate limiting delay to avoid hitting API limits
        time.sleep(0.2)  # 5 requests per second limit
        
        return features
    
    def format_transaction_features(self, features: Dict[str, Any]) -> str:
        """
        Format transaction features into a readable string for the input.
        
        Args:
            features: Dictionary of transaction features
            
        Returns:
            Formatted string describing the transaction patterns
        """
        if not features:
            return "No transaction data available for this address."
            
        formatted_parts = []
        
        # Format key features with descriptions
        if 'total_transactions' in features:
            formatted_parts.append(f"Total transactions: {features['total_transactions']}")
            
        if 'sent_transactions' in features and 'received_transactions' in features:
            formatted_parts.append(f"Sent: {features['sent_transactions']}, Received: {features['received_transactions']}")
            
        if 'unique_to_addresses' in features and 'unique_from_addresses' in features:
            formatted_parts.append(f"Unique recipients: {features['unique_to_addresses']}, Unique senders: {features['unique_from_addresses']}")
            
        if 'total_value' in features:
            formatted_parts.append(f"Total value: {features['total_value']:.6f} ETH")
            
        if 'min_value' in features and 'max_value' in features and 'avg_value' in features:
            formatted_parts.append(f"Value range: {features['min_value']:.6f} - {features['max_value']:.6f} ETH (avg: {features['avg_value']:.6f})")
            
        if 'avg_gas_price' in features:
            formatted_parts.append(f"Average gas price: {features['avg_gas_price']:,.0f} wei")
            
        if 'total_gas_used' in features:
            formatted_parts.append(f"Total gas used: {features['total_gas_used']:,}")
            
        if 'avg_time_between_tx' in features:
            formatted_parts.append(f"Average time between transactions: {features['avg_time_between_tx']:.1f} minutes")
            
        if 'total_time_span' in features:
            hours = features['total_time_span'] / 60
            if hours > 24:
                days = hours / 24
                formatted_parts.append(f"Activity span: {days:.1f} days")
            else:
                formatted_parts.append(f"Activity span: {hours:.1f} hours")
                
        return "; ".join(formatted_parts)
    
    def create_mistral_training_example(self, address_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a single training example in Mistral format.
        
        Args:
            address_data: Dictionary containing address, comment, and date
            
        Returns:
            Training example in Mistral format or None if comment is empty
        """
        address = address_data.get('address', '')
        comment = address_data.get('comment', '').strip()
        date = address_data.get('date', '')
        
        # Skip if comment is empty or address is missing
        if not comment or not address:
            return None
            
        # Get transaction features for this address
        transaction_features = self.get_transaction_features(address)
        
        # Skip if we couldn't get transaction data
        if transaction_features is None:
            return None
            
        transaction_text = self.format_transaction_features(transaction_features)
        
        # Skip if transaction text is empty or indicates no data
        if not transaction_text or transaction_text == "No transaction data available for this address.":
            return None
        
        # Create the input prompt
        input_text = f"""Analyze the following Ethereum address and its transaction patterns:

Address: {address}
Transaction Analysis: {transaction_text}

Based on the transaction patterns and behavior, explain why this address might be flagged as fraudulent or suspicious."""

        # Create the output (the explanation)
        output_text = comment
        
        # Ensure both input and output have content
        if not input_text.strip() or not output_text.strip():
            return None
        
        # Mistral fine-tuning format with messages
        training_example = {
            "messages": [
                {
                    "role": "user",
                    "content": input_text
                },
                {
                    "role": "assistant", 
                    "content": output_text
                }
            ]
        }
        
        return training_example
    
    def transform_dataset(self) -> List[Dict[str, Any]]:
        """
        Transform the entire dataset into Mistral fine-tuning format.
        
        Returns:
            List of training examples in Mistral format
        """
        if self.addresses_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        training_examples = []
        skipped_count = 0
        no_transactions_count = 0
        no_comment_count = 0
        api_error_count = 0
        
        # Use tqdm to show progress
        for address_data in tqdm(self.addresses_data, desc="Processing addresses", unit="address"):
            example = self.create_mistral_training_example(address_data)
            
            if example is None:
                skipped_count += 1
                # Check specific reasons for skipping
                if not address_data.get('comment', '').strip():
                    no_comment_count += 1
                continue
                
            # Validate the example has proper structure
            if 'messages' not in example or len(example['messages']) != 2:
                skipped_count += 1
                continue
                
            # Validate each message has role and content
            valid_example = True
            for message in example['messages']:
                if 'role' not in message or 'content' not in message:
                    valid_example = False
                    break
                if not message['content'].strip():
                    valid_example = False
                    break
                    
            if not valid_example:
                skipped_count += 1
                continue
                
            training_examples.append(example)
        
        # Print summary after completion
        print("\nTransformation complete:")
        print(f"  - Total addresses processed: {len(self.addresses_data)}")
        print(f"  - Training examples created: {len(training_examples)}")
        print(f"  - Skipped (total): {skipped_count}")
        print(f"  - Skipped (no comments): {no_comment_count}")
        print(f"  - Addresses without transaction data: {no_transactions_count}")
        
        return training_examples
    
    def save_dataset(self, training_examples: List[Dict[str, Any]]) -> None:
        """
        Save the training dataset to a JSONL file (one JSON object per line).
        
        Args:
            training_examples: List of training examples to save
        """
        # Update output path to use .jsonl extension
        jsonl_output_path = self.output_path.replace('.json', '.jsonl')
        print(f"Saving {len(training_examples)} training examples to {jsonl_output_path}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(jsonl_output_path), exist_ok=True)
        
        try:
            with open(jsonl_output_path, 'w', encoding='utf-8') as f:
                for example in training_examples:
                    # Write each example as a single line of JSON
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Dataset saved successfully to {jsonl_output_path}!")
        except Exception as e:
            print(f"Error saving dataset: {e}")
            raise
    
    def run(self) -> None:
        """Run the complete transformation pipeline."""
        print("Starting fraud explanation data transformation...")
        
        # Load data
        print("Loading address data...")
        self.load_data()
        print(f"Loaded {len(self.addresses_data)} addresses from {self.addresses_json_path}")
        
        # Transform dataset
        training_examples = self.transform_dataset()
        
        # Save dataset
        self.save_dataset(training_examples)
        
        print("Transformation pipeline completed successfully!")


def main():
    """Main function to run the data transformation."""
    # You can set your Etherscan API key here or pass it as an environment variable
    api_key = os.getenv('ETHERSCAN_API_KEY', 'WT5E1QCQAEGK556D626MCTTNA7W4GERA7T')
    
    if not api_key:
        print("Warning: No Etherscan API key provided. Set ETHERSCAN_API_KEY environment variable or modify the script.")
        print("Transaction features will not be available without an API key.")
    
    transformer = FraudExplanationDataTransformer(etherscan_api_key=api_key)
    transformer.run()


if __name__ == "__main__":
    main()
