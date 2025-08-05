from pymongo import MongoClient

# Replace with your MongoDB URI
client = MongoClient("mongodb://localhost:27017/")  # or MongoDB Atlas URI
db = client["spam_app"]
users_collection = db["users"]
permissions_collection = db["permissions"]
