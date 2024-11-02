import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

data = {
    'user_ID': [1, 1, 2, 2, 3],
    'event_ID': [1, 2, 1, 3, 5],
    'rating': [1, 2, 3, 4, 5],
    'event_duration': [60, 90, 120, 150, 180],
    'event_type': ['music', 'sports', 'music', 'sports', 'music']
}

df = pd.DataFrame(data)

user_item_matrix = df.pivot(index='user_ID', columns='event_ID', values='rating').fillna(0)

# Normalize the data (optional but recommended)
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_scaled)
print(user_similarity)

# Convert the similarity matrix to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print("user similarity df", user_similarity_df)

def get_similar_users(user_id, num_users=2):
    # Get the most similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_users+1]
    return similar_users

def recommend_items(user_id, num_recommendations=2):
    similar_users = get_similar_users(user_id)
    similar_users_ratings = user_item_matrix.loc[similar_users]
    
    # Calculate the average ratings of the similar users for each item
    avg_ratings = similar_users_ratings.mean(axis=0)
    
    # Get the items that the target user has not rated
    user_ratings = user_item_matrix.loc[user_id]
    print("user ratings", user_ratings)
    items_to_recommend = avg_ratings[user_ratings == 0]
    print("items to recommend", items_to_recommend)
    
    # Recommend the top N items
    recommended_items = items_to_recommend.sort_values(ascending=False).head(num_recommendations).index
    print("recommended items", recommended_items)
    return recommended_items

# Example usage
user_id = 1
user_id2 = 2
user_id3 = 3
recommendations = recommend_items(user_id)
print(f"Recommendations for user {user_id}: {recommendations.tolist()}")
print(f"Recommendations for user {user_id2}: {recommendations.tolist()}")
print(f"Recommendations for user {user_id3}: {recommendations.tolist()}")

