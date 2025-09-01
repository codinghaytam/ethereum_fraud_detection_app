import redis
import psycopg2
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import uuid

# Load environment variables early
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Cache configuration
CACHE_EXPIRATION = 3600  # 1 hour in seconds
CACHE_PREFIX = "fraud_detection:"

def create_cache_connection():
    return redis.Redis(host='localhost', port=6379, db=0)

def set_value_in_cache(key, value):
    cache = create_cache_connection()
    cache.set(key, value, ex=CACHE_EXPIRATION)

def get_value_in_cache(key):
    cache = create_cache_connection()
    cached_value = cache.get(key)
    if cached_value:
        return cached_value.decode('utf-8') if isinstance(cached_value, bytes) else cached_value
    return None

def cache_prediction_result(address, prediction_data):
    """Cache fraud detection result for an address"""
    try:
        cache = create_cache_connection()
        cache_key = f"{CACHE_PREFIX}address:{address.lower()}"

        # Add timestamp to cached data
        prediction_data['cached_at'] = datetime.now().isoformat()

        # Store as JSON string with expiration
        cache.set(cache_key, json.dumps(prediction_data), ex=CACHE_EXPIRATION)
        return True
    except Exception as e:
        print(f"Error caching prediction: {e}")
        return False

def get_cached_prediction(address):
    """Get cached fraud detection result for an address"""
    try:
        cache = create_cache_connection()
        cache_key = f"{CACHE_PREFIX}address:{address.lower()}"

        cached_data = cache.get(cache_key)
        if cached_data:
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            return json.loads(cached_data)
        return None
    except Exception as e:
        print(f"Error retrieving cached prediction: {e}")
        return None

def invalidate_prediction_cache(address):
    """Remove cached prediction for an address"""
    try:
        cache = create_cache_connection()
        cache_key = f"{CACHE_PREFIX}address:{address.lower()}"
        cache.delete(cache_key)
        return True
    except Exception as e:
        print(f"Error invalidating cache: {e}")
        return False

def cache_transaction_data(address, transactions):
    """Cache transaction data for an address"""
    try:
        cache = create_cache_connection()
        cache_key = f"{CACHE_PREFIX}transactions:{address.lower()}"

        cache_data = {
            'transactions': transactions,
            'cached_at': datetime.now().isoformat()
        }

        # Store with shorter expiration for transaction data (30 minutes)
        cache.set(cache_key, json.dumps(cache_data), ex=1800)
        return True
    except Exception as e:
        print(f"Error caching transactions: {e}")
        return False

def get_cached_transactions(address):
    """Get cached transaction data for an address"""
    try:
        cache = create_cache_connection()
        cache_key = f"{CACHE_PREFIX}transactions:{address.lower()}"

        cached_data = cache.get(cache_key)
        if cached_data:
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            data = json.loads(cached_data)
            return data.get('transactions', [])
        return None
    except Exception as e:
        print(f"Error retrieving cached transactions: {e}")
        return None

def get_cache_stats():
    """Get Redis cache statistics"""
    try:
        cache = create_cache_connection()
        info = cache.info()

        # Get all keys with our prefix
        pattern = f"{CACHE_PREFIX}*"
        keys = cache.keys(pattern)

        return {
            'redis_info': {
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            },
            'cache_keys_count': len(keys),
            'cache_keys': [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys[:10]]  # Show first 10 keys
        }
    except Exception as e:
        print(f"Error getting cache stats: {e}")
        return {'error': str(e)}

def db_connection():
    conn=psycopg2.connect(
        host='localhost',
        port=5432,
        database='mydb',
        user='postgres',
        password=os.getenv('PG_PASSWORD')
    )
    return conn
def define_prediction_table():
    conn=db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id VARCHAR(255) PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    addresses_involved TEXT[] NOT NULL,
    fraudulent_transactions JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(prediction):
    conn = db_connection()
    cursor = conn.cursor()

    # Generate unique ID for this prediction
    prediction_id = str(uuid.uuid4())

    # Convert addresses_involved to array format for PostgreSQL
    addresses_array = prediction.get('addresses_involved', [])
    if isinstance(addresses_array, set):
        addresses_array = list(addresses_array)

    # Convert fraudulent_transactions to JSON
    fraudulent_transactions_json = json.dumps(prediction.get('fraudulent_transactions', []))

    cursor.execute("""
    INSERT INTO predictions
    (
        id,
        address,
        confidence,
        is_fraud,
        addresses_involved,
        fraudulent_transactions,
        created_at,
        updated_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        prediction_id,
        prediction.get('address'),
        prediction.get('confidence', 0.0),
        prediction.get('is_fraud', False),
        addresses_array,
        fraudulent_transactions_json,
        datetime.now(),
        datetime.now()
    ))

    conn.commit()
    conn.close()

    # Cache the prediction result
    cache_prediction_result(prediction.get('address'), prediction)

    return prediction_id

def update_prediction(prediction_id, prediction):
    conn = db_connection()
    cursor = conn.cursor()

    # Convert addresses_involved to array format for PostgreSQL
    addresses_array = prediction.get('addresses_involved', [])
    if isinstance(addresses_array, set):
        addresses_array = list(addresses_array)

    # Convert fraudulent_transactions to JSON
    fraudulent_transactions_json = json.dumps(prediction.get('fraudulent_transactions', []))

    cursor.execute("""
    UPDATE predictions
    SET 
        confidence = %s,
        is_fraud = %s,
        addresses_involved = %s,
        fraudulent_transactions = %s,
        updated_at = %s
    WHERE id = %s
    """, (
        prediction.get('confidence', 0.0),
        prediction.get('is_fraud', False),
        addresses_array,
        fraudulent_transactions_json,
        datetime.now(),
        prediction_id
    ))

    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()

    # Update cache if update was successful
    if rows_affected > 0:
        cache_prediction_result(prediction.get('address'), prediction)

    return rows_affected > 0

def get_prediction_by_address(address):
    conn = db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, address, confidence, is_fraud, addresses_involved, 
           fraudulent_transactions, created_at, updated_at
    FROM predictions
    WHERE address = %s
    ORDER BY updated_at DESC
    LIMIT 1
    """, (address,))

    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            'id': result[0],
            'address': result[1],
            'confidence': result[2],
            'is_fraud': result[3],
            'addresses_involved': result[4],
            'fraudulent_transactions': json.loads(result[5]) if result[5] else [],
            'created_at': result[6],
            'updated_at': result[7]
        }
    return None

def get_all_predictions():
    conn = db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, address, confidence, is_fraud, addresses_involved, 
           fraudulent_transactions, created_at, updated_at
    FROM predictions
    ORDER BY updated_at DESC
    """)

    results = cursor.fetchall()
    conn.close()

    predictions = []
    for result in results:
        predictions.append({
            'id': result[0],
            'address': result[1],
            'confidence': result[2],
            'is_fraud': result[3],
            'addresses_involved': result[4],
            'fraudulent_transactions': json.loads(result[5]) if result[5] else [],
            'created_at': result[6],
            'updated_at': result[7]
        })

    return predictions
