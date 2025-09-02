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
    # Connect to redis service in docker network
    return redis.Redis(host='redis', port=6379, db=0)

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
    # Connect to postgres service in docker network; only secret required is PG_PASSWORD
    conn = psycopg2.connect(
        host='db',
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

    # Cache the prediction result (ensure prediction_id is present in cached payload)
    prediction_to_cache = { **prediction, 'prediction_id': prediction_id }
    cache_prediction_result(prediction.get('address'), prediction_to_cache)

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

    # Update cache if update was successful (ensure prediction_id is present)
    if rows_affected > 0:
        prediction_to_cache = { **prediction, 'prediction_id': prediction_id }
        cache_prediction_result(prediction.get('address'), prediction_to_cache)

    return rows_affected > 0

def _parse_json_field(value, default=None):
    """Safely parse a JSON/JSONB field regardless of driver return type.
    - If value is str, parse with json.loads.
    - If value is list/dict, return as-is.
    - None -> default (list by default).
    """
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    if isinstance(value, (list, dict)):
        return value
    return default

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
            'fraudulent_transactions': _parse_json_field(result[5], []),
            'created_at': result[6],
            'updated_at': result[7]
        }
    return None

def get_prediction_by_id(prediction_id: str):
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, address, confidence, is_fraud, addresses_involved,
               fraudulent_transactions, created_at, updated_at
        FROM predictions
        WHERE id = %s
        LIMIT 1
        """,
        (prediction_id,),
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            'id': result[0],
            'address': result[1],
            'confidence': result[2],
            'is_fraud': result[3],
            'addresses_involved': result[4],
            'fraudulent_transactions': _parse_json_field(result[5], []),
            'created_at': result[6],
            'updated_at': result[7],
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
            'fraudulent_transactions': _parse_json_field(result[5], []),
            'created_at': result[6],
            'updated_at': result[7]
        })

    return predictions

def define_prediction_reports_table():
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_reports (
        id VARCHAR(255) PRIMARY KEY,
        prediction_id VARCHAR(255) NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL,
        is_valid BOOLEAN NOT NULL,
        note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (prediction_id, user_id)
    )
    ''')
    conn.commit()
    conn.close()


def insert_prediction_report(prediction_id: str, user_id: str, is_valid: bool, note: str = None) -> str:
    """Create or update a user's report for a prediction. Returns report row id."""
    conn = db_connection()
    cursor = conn.cursor()
    report_id = str(uuid.uuid4())
    cursor.execute(
        '''
        INSERT INTO prediction_reports (id, prediction_id, user_id, is_valid, note, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (prediction_id, user_id)
        DO UPDATE SET is_valid = EXCLUDED.is_valid, note = EXCLUDED.note, created_at = EXCLUDED.created_at
        RETURNING id
        ''',
        (
            report_id,
            prediction_id,
            user_id,
            is_valid,
            note,
            datetime.now(),
        ),
    )
    upserted_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return upserted_id


def get_reports_for_prediction(prediction_id: str):
    """Return all reports for a prediction (without PII filtering)."""
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT user_id, is_valid, note, created_at
        FROM prediction_reports
        WHERE prediction_id = %s
        ORDER BY created_at DESC
        ''',
        (prediction_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            'user_id': r[0],
            'is_valid': r[1],
            'note': r[2],
            'created_at': r[3].isoformat() if r[3] else None,
        }
        for r in rows
    ]


def get_report_stats_for_prediction(prediction_id: str):
    """Return aggregated counts of valid/invalid reports for a prediction."""
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT
            COALESCE(SUM(CASE WHEN is_valid THEN 1 ELSE 0 END), 0) AS valid_count,
            COALESCE(SUM(CASE WHEN NOT is_valid THEN 1 ELSE 0 END), 0) AS invalid_count,
            COUNT(*) AS total_count
        FROM prediction_reports
        WHERE prediction_id = %s
        ''',
        (prediction_id,),
    )
    row = cursor.fetchone()
    conn.close()
    valid_count = row[0] if row and row[0] is not None else 0
    invalid_count = row[1] if row and row[1] is not None else 0
    total_count = row[2] if row and row[2] is not None else 0
    return {
        'valid_count': int(valid_count),
        'invalid_count': int(invalid_count),
        'total_count': int(total_count),
    }
