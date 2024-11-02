from user import create_user

user_data = {"name": "John Doe", "email": "john.doe@example.com", "age": 30}

user_id = create_user(user_data)
print(f"User added with ID: {user_id}")
