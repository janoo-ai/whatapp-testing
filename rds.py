import pymysql
from dbutils.pooled_db import PooledDB


# Check if required environment variables are loaded
DB_HOST = "jaano-db.c5ywcqqaibiq.ap-south-1.rds.amazonaws.com"
DB_USER = "jaanoadmin"
DB_PASSWORD = "t8poeVmIPvmrv134mzVb"
DB_NAME = "jaanodb"

# Setup the connection pool
try:
    pool = PooledDB(
        creator=pymysql,  # The creator is pymysql for MySQL
        maxconnections=3,  # Maximum number of connections in the pool
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True  # Enable auto-commit for each connection
    )
    print("Database connection pool initialized successfully.")
except Exception as e:
    print(f"Error initializing connection pool: {e}")
    pool = None

# Function to get a connection from the pool
def get_db_connection():
    if pool:
        try:
            con = pool.connection()
            return con
        except Exception as e:
            print(f"Error getting connection from pool: {e}")
            return None
    else:
        print("Database pool is not initialized.")
        return None
