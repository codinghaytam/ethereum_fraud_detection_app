#!/bin/bash
set -e

echo "🚀 Starting FastAPI Fraud Detection Service..."

# Set proper Python path
export PYTHONPATH="/app:${PYTHONPATH}"

# Set NumPy configuration for Docker environment
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Check if required environment variables are set
if [ -z "$ETHERSCAN_API_KEY" ]; then
    echo "❌ ETHERSCAN_API_KEY environment variable is not set"
    exit 1
fi

# Clear any Python cache that might cause issues
find /app -name "*.pyc" -delete
find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Test Python imports before starting the server
echo "🔍 Testing Python imports..."
python3 -c "
import sys
import os

# Add some debugging info
print('Python version:', sys.version)
print('Python executable:', sys.executable)
print('Python path:', sys.path[:3])

try:
    # Import NumPy with detailed error handling
    print('Testing NumPy import...')
    try:
        import numpy as np
        print('✅ NumPy version:', np.__version__)
        print('NumPy installation path:', np.__file__)
        
        # Test basic NumPy functionality
        test_array = np.array([1, 2, 3])
        print('✅ NumPy array test successful:', test_array)
        
    except ImportError as np_error:
        print(f'❌ NumPy import error: {np_error}')
        print('Attempting to diagnose NumPy installation...')
        
        # Try to get more information about the error
        import importlib.util
        spec = importlib.util.find_spec('numpy')
        if spec is None:
            print('NumPy package not found')
        else:
            print(f'NumPy found at: {spec.origin}')
        
        # Try reinstalling NumPy
        print('Attempting to reinstall NumPy...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', 'numpy==1.24.4'])
        import numpy as np
        print('✅ NumPy reinstalled, version:', np.__version__)
    
    # Test PyTorch import
    print('Testing PyTorch import...')
    import torch
    print('✅ PyTorch version:', torch.__version__)
    print('PyTorch installation path:', torch.__file__)
    
    # Test other dependencies
    import pandas as pd
    print('✅ Pandas version:', pd.__version__)
    
    import sklearn
    print('✅ Scikit-learn version:', sklearn.__version__)
    
    import fastapi
    print('✅ FastAPI imported successfully')
    
    print('✅ All imports successful!')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed. Exiting..."
    exit 1
fi

echo "✅ All imports successful. Starting server..."

# Change to the source directory
cd /app

# Start the FastAPI server with proper module path
exec uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
