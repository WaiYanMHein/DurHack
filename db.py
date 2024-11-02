from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Connect to MongoDB
load_dotenv()  # Load environment variables from .env file
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)  # Replace with your MongoDB URI
db = client.get_database()  # Replace with your database name


def check_connection():
    try:
        # The ismaster command is cheap and does not require auth.
        client.admin.command("ismaster")
        print("MongoDB connection successful")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")


# Call the function to check the connection
check_connection()
