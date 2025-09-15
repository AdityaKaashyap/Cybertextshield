from pymongo import MongoClient

MONGO_URI = "mongodb+srv://adityapk333_db_user:AObQAgBBMJJo6GmF@smish-collector.zaxj8tc.mongodb.net/?retryWrites=true&w=majority&appName=Smish-collector"
client = MongoClient(MONGO_URI)
db = client["cybertextshield"]

users_collection = db["users"]
permissions_collection = db["permissions"]
ham_messages_collection = db["ham_messages"]
smish_messages_collection = db["smish_messages"]
