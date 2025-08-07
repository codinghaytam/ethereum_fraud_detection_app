#!/bin/bash
set -e

echo "🚀 Starting FastAPI Fraud Detection Service..."

# Set proper Python path
export PYTHONPATH="/app:${PYTHONPATH}"

# Check if required environment variables are set
if [ -z "$ETHERSCAN_API_KEY" ]; then
    echo "❌ ETHERSCAN_API_KEY environment variable is not set"
    exit 1
fi

# Test Python imports before starting the server
echo "🔍 Testing Python imports..."
python3 -c "
import sys
import os
try:
    # Try importing NumPy first with specific error handling
    try:
        import numpy as np
        print('✅ NumPy version:', np.__version__)
    except ImportError as np_error:
        print(f'❌ NumPy import error: {np_error}')
        print('Attempting to reinstall NumPy...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'numpy==1.24.4'])
        import numpy as np
        print('✅ NumPy reinstalled, version:', np.__version__)
    
    import pandas as pd
    print('✅ Pandas version:', pd.__version__)
    
    import torch
    print('✅ PyTorch version:', torch.__version__)
    
    import sklearn
    print('✅ Scikit-learn version:', sklearn.__version__)
    
    import fastapi
    print('✅ FastAPI imported successfully')
    
    print('✅ All imports successful!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed. Exiting..."
    exit 1
fi

echo "✅ All imports successful. Starting server..."

# Start the FastAPI server
exec uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
