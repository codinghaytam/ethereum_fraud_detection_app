from fastapi import FastAPI, HTTPException
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from fastapi.testclient import TestClient


# Import scientific libraries first to avoid conflicts
import numpy as np
import pandas as pd
from src.FraudTransactionDetector import detect_fraud
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://3.142.201.165:3000"],  # Specify the exact frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   
# Load environment variables early
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Import after setting up environment and path
import requests
from src.modelLoader import predict_address_fraud

# Load model using the new generic approach


app = FastAPI()

# Get Etherscan API key from environment variable
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

if not ETHERSCAN_API_KEY:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")

def fetch_transactions_from_etherscan(address: str, api_key: str = None):
    """
    Fetch normal transactions from Etherscan API for a given address
    
    Args:
        address (str): Ethereum address to fetch transactions for
        api_key (str): Etherscan API key (optional, uses default if not provided)
    
    Returns:
        list: List of transaction dictionaries
    """
    if api_key is None:
        api_key = ETHERSCAN_API_KEY
    
    url = "https://api.etherscan.io/api"
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'offset': 1000,  # Limit to 1000 transactions
        'page': 1,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == '1':
            return data['result']
        else:
            print(f"Etherscan API error: {data.get('message', 'Unknown error')}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


@app.post("/api/processAdress")
async def process_addresses(address: str):
   
    try:
            # Use absolute path resolution for model directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, '..', 'model')+'/address_fraud_classifier_lstm.pth'

            # Fetch transactions from Etherscan
            transactions = fetch_transactions_from_etherscan(address, ETHERSCAN_API_KEY)
            
            # Get fraud transaction analysis
            fraud_transactions = detect_fraud(
                address=address,
                api_key=ETHERSCAN_API_KEY,
                transactions=fetch_transactions_from_etherscan(address, ETHERSCAN_API_KEY)
            )
            
            return dict(fraud_transactions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/")
async def root():
    return {"message": "Ethereum Address Processor API", "status": "running"}

@app.get("/debug/model")
async def debug_model():
    """Debug endpoint to check model file availability"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '..', 'model')
    
    debug_info = {
        "script_dir": script_dir,
        "model_dir_absolute": os.path.abspath(model_dir),
        "model_dir_exists": os.path.exists(model_dir),
        "model_files": []
    }
    
    if os.path.exists(model_dir):
        debug_info["model_files"] = os.listdir(model_dir)
    else:
        # Try relative path
        relative_model_dir = '../model/'
        debug_info["relative_model_dir_exists"] = os.path.exists(relative_model_dir)
        if os.path.exists(relative_model_dir):
            debug_info["relative_model_files"] = os.listdir(relative_model_dir)
    
    return debug_info


client = TestClient(app)

def test_address_analysis():
    """Test the address analysis endpoint"""
    test_address = "0x742d35c68a8e8c9b6f8e9e15e7f8a5e3d2b1c0a9"  # Example address
    response = client.post("/api/processAdress",params={"address": test_address})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}, response: {response.text}"
    data = response.json()
    