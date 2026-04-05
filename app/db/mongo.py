from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["rakr_db"]
users_collection = db["users"]
chats_collection = db["chats"]        # { _id, google_id, title, created_at, updated_at }
messages_collection = db["messages"]  # { _id, chat_id, role, text, sources, context, eval, created_at }
