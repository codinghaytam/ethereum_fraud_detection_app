from fastapi import FastAPI, HTTPException
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get Etherscan API key from environment variable
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
print(ETHERSCAN_API_KEY)

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
            
            return {
                "address": addresse,
                "status": "success",
                "transaction_count": len(transactions),
                "transactions": transactions
            }
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