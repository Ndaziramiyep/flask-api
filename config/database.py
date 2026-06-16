import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'teascan_ai')

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=10000,
    socketTimeoutMS=30000,
)
db = client[DB_NAME]

# Collections
users_col = db['users']
detections_col = db['detections']
disease_info_col = db['disease_information']
notifications_col = db['notifications']


def create_indexes():
    """Create MongoDB indexes. Safe to call multiple times (idempotent)."""
    try:
        users_col.create_index('email', unique=True)

        detections_col.create_index([('userId', ASCENDING), ('createdAt', DESCENDING)])
        detections_col.create_index('userId')

        notifications_col.create_index([('userId', ASCENDING), ('isRead', ASCENDING)])
        notifications_col.create_index('userId')

        print('MongoDB indexes created/verified successfully')
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f'Warning: Could not create indexes — MongoDB unreachable: {e}')
    except Exception as e:
        print(f'Warning: Index creation error: {e}')


def ping_db():
    """Verify Atlas connection at startup."""
    try:
        client.admin.command('ping')
        print(f'MongoDB Atlas connected — database: {DB_NAME}')
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f'MongoDB Atlas connection failed: {e}')
        return False
