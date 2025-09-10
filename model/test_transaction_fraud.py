#!/usr/bin/env python3
"""
Test script for transaction-level fraud detection
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.getcwd())

from fraud_detector import SimpleFraudDetector

def test_transaction_level_fraud_detection():
    """Test the new transaction-level fraud detection"""
    
    # Initialize detector
    detector = SimpleFraudDetector()
    
    # Test address (you can replace with any Ethereum address)
    test_address = "0x742d35c68a8e8c9b6f8e9e15e7f8a5e3d2b1c0a9"  # Example address
    
    # API key (you'll need to replace with your actual Etherscan API key)
    api_key = "YourEtherscanAPIKey"
    
    print("=" * 60)
    print("TRANSACTION-LEVEL FRAUD DETECTION TEST")
    print("=" * 60)
    print(f"Testing address: {test_address}")
    print(f"Analyzing transactions for fraudulent patterns...")
    print()
    
    try:
        # Run the new transaction-level analysis
        result = detector.predict(test_address, api_key, count=10)
        
        print("RESULTS:")
        print("-" * 40)
        print(f"Found {len(result)} fraudulent transactions with confidence > 0.5")
        print()
        
        if result:
            print("FRAUDULENT TRANSACTIONS (ranked by confidence):")
            print("-" * 50)
            
            for i, tx in enumerate(result, 1):
                print(f"{i}. Address: {tx['address']}")
                print(f"   Confidence: {tx['confidence']:.3f}")
                print()
        else:
            print("✅ No high-confidence fraudulent transactions found!")
            print("This address appears to have legitimate transaction patterns.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have a valid Etherscan API key")
        print("2. Check that the address format is correct")
        print("3. Verify the model file 'gradient_boosting_model_random_tree.pkl' exists")
        print("4. Make sure all required packages are installed")

def demo_with_sample_data():
    """Demo function showing expected output format"""
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT FORMAT DEMO")
    print("=" * 60)
    
    # Sample simplified result format
    sample_result = [
        {
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "confidence": 0.87
        },
        {
            "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "confidence": 0.73
        },
        {
            "address": "0x9876543210fedcba9876543210fedcba98765432", 
            "confidence": 0.65
        }
    ]
    
    print("SAMPLE RESULTS:")
    print("-" * 40)
    print(f"Found {len(sample_result)} fraudulent transactions with confidence > 0.5")
    print()
    
    print("FRAUDULENT TRANSACTIONS (ranked by confidence):")
    print("-" * 50)
    
    for i, tx in enumerate(sample_result, 1):
        print(f"{i}. Address: {tx['address']}")
        print(f"   Confidence: {tx['confidence']:.3f}")
        print()

if __name__ == "__main__":
    print("ETHEREUM TRANSACTION-LEVEL FRAUD DETECTOR")
    print("This tool analyzes individual transactions to identify fraudulent addresses")
    print("by confidence level, rather than just classifying the entire address.")
    print()
    
    # Show sample output format
    demo_with_sample_data()
    
    # Uncomment the line below to test with real data (requires API key)
    # test_transaction_level_fraud_detection()
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS:")
    print("- ✅ Transaction-level analysis instead of address-level")
    print("- ✅ Simple output: only address and confidence")
    print("- ✅ Confidence ranking of fraudulent transactions")
    print("- ✅ Configurable result count (default: top 10)")
    print("- ✅ Confidence threshold filtering (>0.5)")
    print("- ✅ Returns empty list if no fraudulent transactions found")
    print("=" * 60)
