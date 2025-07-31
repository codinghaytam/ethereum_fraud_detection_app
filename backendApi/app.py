from fastapi import FastAPI, HTTPException
import requests
import os
from dotenv import load_dotenv
from modelLoader import load_model, prepare_etherscan_transactions, predict_fraud_probability
import pandas as pd
import numpy as np
import torch
from datetime import datetime




model,static_scaler, sequence_scaler, static_feature_cols, sequence_features, config = load_model("model/fraud_classifier.pth")

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get Etherscan API key from environment variable
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

if not ETHERSCAN_API_KEY:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")


@app.post("/api/processAdresse")
async def process_addresses(addresse: str):
    """
    Process Ethereum addresses using Etherscan API - Get normal transactions only
    """
    try:
        base_url = "https://api.etherscan.io/api"
        
        # Get normal transactions for the address
        params = {
            "module": "account",
            "action": "txlist",
            "address": addresse,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 10000,  # Maximum allowed by Etherscan
            "sort": "desc",    # Latest first
            "apikey": ETHERSCAN_API_KEY
        }
        
        # Make request to Etherscan API
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") == "1":
            transactions = data.get("result", [])
            
            if not transactions:
                return {
                    "address": addresse,
                    "transaction_count": 0,
                    "prediction": {
                        "is_fraud": False,
                        "fraud_probability": 0.0,
                        "non_fraud_probability": 1.0,
                        "confidence": 1.0,
                        "status": "success",
                        "message": "No transactions found for this address"
                    },
                    "processing_info": {
                        "etherscan_api_status": "success",
                        "feature_extraction_status": "skipped",
                        "prediction_status": "success"
                    }
                }
            
            # Process transactions into model features
            print(f"Processing {len(transactions)} transactions for address {addresse}")
            sequence_data, static_data = prepare_etherscan_transactions(
                transactions, sequence_features, static_feature_cols
            )
            
            if sequence_data is None:
                return {
                    "address": addresse,
                    "transaction_count": len(transactions),
                    "prediction": {
                        "is_fraud": False,
                        "fraud_probability": 0.0,
                        "non_fraud_probability": 1.0,
                        "confidence": 0.0,
                        "status": "error",
                        "message": "Unable to process transaction features"
                    },
                    "processing_info": {
                        "etherscan_api_status": "success",
                        "feature_extraction_status": "failed",
                        "prediction_status": "failed"
                    }
                }
            
            # Make fraud prediction
            print(f"Making fraud prediction for address {addresse}")
            prediction = predict_fraud_probability(
                model, sequence_data, static_data, sequence_scaler, static_scaler
            )
            
            # Prepare response with comprehensive information
            response = {
                "address": addresse,
                "prediction": prediction,
                
                "processing_info": {
                    "etherscan_api_status": "success",
                    "feature_extraction_status": "success" if sequence_data is not None else "failed",
                    "prediction_status": prediction.get('status', 'unknown'),
                    "transactions_processed": len(transactions),
                    "sequence_shape": list(sequence_data.shape) if sequence_data is not None else None,
                    "static_features_count": len(static_data[0]) if static_data is not None else 0
                }
            }
            
            print(f"Prediction completed for address {addresse}: {'FRAUD' if prediction.get('is_fraud') else 'NOT FRAUD'} (confidence: {prediction.get('confidence', 0):.3f})")
            
            return response
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Etherscan API error: {data.get('message', 'Unknown error')}")
            
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

@app.get("/")
async def root():
    return {"message": "Ethereum Address Processor API", "status": "running"}

