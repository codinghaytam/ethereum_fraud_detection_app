from fastapi import FastAPI, HTTPException
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Import scientific libraries first to avoid conflicts
import numpy as np
import pandas as pd
import torch

# Load environment variables early
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Add the parent directory to Python path to find modelLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import after setting up environment and path
import requests
from modelLoader import predict_address_state

# Load model using the new generic approach


app = FastAPI()

# Get Etherscan API key from environment variable
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

if not ETHERSCAN_API_KEY:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")


@app.post("/api/processAdress")
async def process_addresses(address: str):
   
    try:
            # Use absolute path resolution for model directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, '..', 'model')
            
            # Fallback to relative path if absolute doesn't work
            if not os.path.exists(model_dir):
                model_dir = '../model/'
            
            prediction = predict_address_state(
                address=address, 
                apikey=ETHERSCAN_API_KEY, 
                modelDir=model_dir
            )
            return {
                "address": address,
                "prediction": prediction["result"],
                "transactions_used": prediction["transactionsUsed"],
            }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

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


