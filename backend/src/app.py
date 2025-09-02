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
from src.db import (
    insert_prediction,
    update_prediction,
    get_prediction_by_address,
    get_all_predictions,
    define_prediction_table,
    get_cached_prediction,
    cache_prediction_result,
    invalidate_prediction_cache,
    get_cached_transactions,
    cache_transaction_data,
    get_cache_stats,
    # new imports for reporting
    define_prediction_reports_table,
    insert_prediction_report,
    get_reports_for_prediction,
    get_report_stats_for_prediction,
    get_prediction_by_id,
)


# Load environment variables early
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Import after setting up environment and path
import requests


# Load model using the new generic approach


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://3.142.201.165:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Get Etherscan API key from environment variable
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

if not ETHERSCAN_API_KEY:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")

def fetch_transactions_from_etherscan(address: str, api_key: str = None):
    """
    Fetch normal transactions from Etherscan API for a given address with caching

    Args:
        address (str): Ethereum address to fetch transactions for
        api_key (str): Etherscan API key (optional, uses default if not provided)
    
    Returns:
        list: List of transaction dictionaries
    """
    if api_key is None:
        api_key = ETHERSCAN_API_KEY
    
    # Check cache first
    cached_transactions = get_cached_transactions(address)
    if cached_transactions:
        print(f"Using cached transactions for address: {address}")
        return cached_transactions

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
            transactions = data['result']
            # Cache the transaction data
            cache_transaction_data(address, transactions)
            print(f"Fetched and cached {len(transactions)} transactions for address: {address}")
            return transactions
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
        # Initialize database table if not exists
        define_prediction_table()

        # Check cache first for existing prediction
        cached_prediction = get_cached_prediction(address)
        if cached_prediction:
            print(f"Using cached prediction for address: {address}")
            cached_prediction['source'] = 'cache'
            cached_prediction['database_action'] = 'cached'
            return cached_prediction

        # Use absolute path resolution for model directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, '..', 'model')+'/address_fraud_classifier_lstm.pth'

        # Fetch transactions from Etherscan (with caching)
        transactions = fetch_transactions_from_etherscan(address, ETHERSCAN_API_KEY)

        # Get fraud transaction analysis
        fraud_analysis = detect_fraud(
            address=address,
            api_key=ETHERSCAN_API_KEY,
            transactions=transactions
        )

        # Prepare prediction data for database
        prediction_data = {
            'address': address,
            'confidence': fraud_analysis.get('confidence', 0.0),
            'is_fraud': fraud_analysis.get('is_fraud', False),
            'addresses_involved': fraud_analysis.get('addresses_involved', []),
            'fraudulent_transactions': fraud_analysis.get('fraudulent_transactions', [])
        }

        # Check if prediction for this address already exists in database
        existing_prediction = get_prediction_by_address(address)

        if existing_prediction:
            # Update existing prediction
            update_success = update_prediction(existing_prediction['id'], prediction_data)
            if update_success:
                fraud_analysis['database_action'] = 'updated'
                fraud_analysis['prediction_id'] = existing_prediction['id']
                fraud_analysis['source'] = 'fresh_analysis'
            else:
                fraud_analysis['database_action'] = 'update_failed'
        else:
            # Insert new prediction
            prediction_id = insert_prediction(prediction_data)
            fraud_analysis['database_action'] = 'inserted'
            fraud_analysis['prediction_id'] = prediction_id
            fraud_analysis['source'] = 'fresh_analysis'

        return fraud_analysis

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

@app.get("/api/predictions")
async def get_predictions():
    """Get all predictions from the database"""
    try:
        predictions = get_all_predictions()
        return {
            "predictions": predictions,
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {e}")

@app.get("/api/predictions/{address}")
async def get_prediction_for_address(address: str):
    """Get the latest prediction for a specific address"""
    try:
        prediction = get_prediction_by_address(address)
        if prediction:
            return prediction
        else:
            raise HTTPException(status_code=404, detail="No prediction found for this address")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prediction: {e}")

@app.get("/api/cache/stats")
async def get_cache_statistics():
    """Get Redis cache statistics and performance metrics"""
    try:
        stats = get_cache_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cache stats: {e}")

@app.delete("/api/cache/predictions/{address}")
async def invalidate_address_cache(address: str):
    """Invalidate cached prediction for a specific address"""
    try:
        success = invalidate_prediction_cache(address)
        if success:
            return {"message": f"Cache invalidated for address: {address}"}
        else:
            return {"message": f"No cache found or error invalidating cache for address: {address}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invalidating cache: {e}")

@app.post("/api/cache/refresh/{address}")
async def refresh_address_cache(address: str):
    """Force refresh cache for a specific address by invalidating and reprocessing"""
    try:
        # Invalidate existing cache
        invalidate_prediction_cache(address)

        # Reprocess the address (this will create fresh cache)
        result = await process_addresses(address)
        result['cache_action'] = 'refreshed'

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing cache: {e}")

@app.on_event("startup")
async def startup_init():
    # Ensure required tables exist
    try:
        define_prediction_table()
        define_prediction_reports_table()
    except Exception as e:
        # Log but don't crash startup; endpoints will raise proper errors later
        print(f"Startup init error: {e}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class ReportRequest(BaseModel):
    prediction_id: str
    user_id: str
    is_valid: bool
    note: str | None = None

@app.post("/api/reports")
async def submit_report(report: ReportRequest):
    try:
        # Validate prediction exists
        pred = get_prediction_by_id(report.prediction_id)
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")

        report_id = insert_prediction_report(
            prediction_id=report.prediction_id,
            user_id=report.user_id,
            is_valid=report.is_valid,
            note=report.note,
        )
        stats = get_report_stats_for_prediction(report.prediction_id)
        return {"report_id": report_id, "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting report: {e}")

@app.get("/api/reports/{prediction_id}")
async def get_reports(prediction_id: str):
    try:
        # Validate prediction exists
        pred = get_prediction_by_id(prediction_id)
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")

        stats = get_report_stats_for_prediction(prediction_id)
        reports = get_reports_for_prediction(prediction_id)
        return {"prediction_id": prediction_id, "stats": stats, "reports": reports}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching reports: {e}")

client = TestClient(app)

def test_address_analysis():
    """Test the address analysis endpoint"""
    test_address = "0x742d35c68a8e8c9b6f8e9e15e7f8a5e3d2b1c0a9"  # Example address
    response = client.post("/api/processAdress",params={"address": test_address})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}, response: {response.text}"
    data = response.json()
