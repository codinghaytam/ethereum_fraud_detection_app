from fastapi import FastAPI, HTTPException
import requests
import os
from dotenv import load_dotenv
import sys
from dotenv import load_dotenv
# Add the parent directory to Python path to find modelLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from modelLoader import predict_address_state
import pandas as pd
import numpy as np
import torch
from datetime import datetime


# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Load model using the new generic approach


app = FastAPI()

# Get Etherscan API key from environment variable
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

if not ETHERSCAN_API_KEY:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")


@app.post("/api/processAdresse")
async def process_addresses(address: str):
   
    try:
            prediction = predict_address_state(
                address=address, 
                apikey=ETHERSCAN_API_KEY, 
                modelDir='../model/'
            )
            return {
                "address": address,
                "prediction": prediction
            }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

@app.get("/")
async def root():
    return {"message": "Ethereum Address Processor API", "status": "running"}


