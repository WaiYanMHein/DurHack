from db import db

users_collection = db["users"]


def create_user(user_data):
    result = users_collection.insert_one(user_data)
    return result.inserted_id


def get_user(user_id):
    user = users_collection.find_one({"_id": user_id})
    return user
