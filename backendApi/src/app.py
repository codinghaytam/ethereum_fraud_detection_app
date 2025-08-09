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
            prediction = predict_address_state(
                address=address, 
                apikey=ETHERSCAN_API_KEY, 
                modelDir= os.path.abspath(__file__)+'../model/'
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


