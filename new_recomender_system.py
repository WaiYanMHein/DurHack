import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------

# Load datasets
df_events = pd.read_csv("events_durham.csv")
df_users_past = pd.read_csv("users_past_durham.csv")
df_users_going = pd.read_csv("users_going_durham.csv")  # Not used in current recommendations

# Convert date columns to datetime
df_events['start_datetime'] = pd.to_datetime(df_events['start_datetime'], errors='coerce')
df_events['end_datetime'] = pd.to_datetime(df_events['end_datetime'], errors='coerce')

# Define cutoff date for splitting past and future events
cutoff_date = datetime(2024, 11, 2)

# Split events into past and future
df_events_past = df_events[df_events['start_datetime'] < cutoff_date].copy()
df_events_future = df_events[df_events['start_datetime'] >= cutoff_date].copy()

# Filter user past interactions to include only past events
df_users_past = df_users_past[df_users_past['event_id'].isin(df_events_past['event_id'])].copy()

# Ensure 'going' column is integer type (if needed in future)
df_users_going['going'] = df_users_going['going'].astype(int)

# Ensure consistent data types
df_users_past['user_id'] = df_users_past['user_id'].astype(int)
df_users_past['event_id'] = df_users_past['event_id'].astype(int)
df_users_past['rating'] = df_users_past['rating'].astype(float)

# ---------------------------
# 2. Collaborative Filtering Using SVD
# ---------------------------

# Create user-item interaction matrix
user_item_matrix = df_users_past.pivot_table(
    index='user_id',
    columns='event_id',
    values='rating'
).fillna(0)

# Normalize the user-item matrix by subtracting the mean rating for each user
user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
user_item_matrix_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

# Perform Singular Value Decomposition (SVD)
k = 50  # Number of latent factors; adjust based on dataset size and performance
U, sigma, Vt = svds(user_item_matrix_demeaned, k=k)
sigma = np.diag(sigma)

# Reconstruct the user-item matrix with predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# Clip the predicted ratings to the valid rating range
min_rating = df_users_past['rating'].min()
max_rating = df_users_past['rating'].max()
all_user_predicted_ratings = np.clip(all_user_predicted_ratings, min_rating, max_rating)

# Convert the predicted ratings to a DataFrame
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# ---------------------------
# 3. Computing User Similarity
# ---------------------------

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# ---------------------------
# 4. Recommendation Functions
# ---------------------------

def get_top_n_similar_users(user_id, user_similarity_df, n=5):
    """
    Retrieve the top-N users most similar to the given user.
    """
    if user_id not in user_similarity_df.index:
        return []
    user_similarities = user_similarity_df.loc[user_id].drop(user_id)
    top_n_users = user_similarities.sort_values(ascending=False).head(n).index.tolist()
    return top_n_users

def get_events_from_similar_users(similar_users, df_users_past, min_rating=4):
    """
    Get a set of events that similar users have highly rated.
    """
    similar_users_ratings = df_users_past[df_users_past['user_id'].isin(similar_users)]
    highly_rated_events = similar_users_ratings[similar_users_ratings['rating'] >= min_rating]['event_id'].tolist()
    return set(highly_rated_events)

def recommend_popular_events(df_events_future, df_users_past, num_recommendations=5):
    """
    Recommend the most popular future events based on past interactions.
    """
    event_popularity = df_users_past.groupby('event_id')['rating'].count().reset_index(name='count')
    popular_events = df_events_future.merge(event_popularity, on='event_id', how='left')
    popular_events['count'] = popular_events['count'].fillna(0)
    popular_events['PredictedRating'] = 0  # Assign a default PredictedRating
    popular_events = popular_events.sort_values('count', ascending=False)
    return popular_events[['event_id', 'name', 'type', 'start_datetime', 'PredictedRating']].head(num_recommendations)

def recommend_events_based_on_similar_users(user_id, df_users_past, df_events_future, user_similarity_df, n_similar_users=5, n_recommendations=5):
    """
    Recommend events to a user based on events highly rated by similar users.
    """
    # Get top-N similar users
    similar_users = get_top_n_similar_users(user_id, user_similarity_df, n=n_similar_users)
    if not similar_users:
        return recommend_popular_events(df_events_future, df_users_past, n_recommendations)
    
    # Get events highly rated by similar users
    similar_users_events = get_events_from_similar_users(similar_users, df_users_past)
    if not similar_users_events:
        return recommend_popular_events(df_events_future, df_users_past, n_recommendations)
    
    # Get events the user has already rated
    user_events = set(df_users_past[df_users_past['user_id'] == user_id]['event_id'].tolist())
    
    # Recommend events not yet rated by the user and available in future events
    recommended_events = similar_users_events - user_events
    recommended_events = recommended_events.intersection(set(df_events_future['event_id'].tolist()))
    
    # Get event details
    recommended_events_df = df_events_future[df_events_future['event_id'].isin(recommended_events)]
    
    # If more events than needed, select top-N based on popularity among similar users
    if len(recommended_events_df) > n_recommendations:
        event_popularity = df_users_past[
            df_users_past['event_id'].isin(recommended_events) & 
            df_users_past['user_id'].isin(similar_users)
        ]
        event_popularity = event_popularity.groupby('event_id')['rating'].count().reset_index(name='count')
        recommended_events_df = recommended_events_df.merge(event_popularity, on='event_id')
        recommended_events_df = recommended_events_df.sort_values('count', ascending=False)
        recommended_events_df = recommended_events_df.head(n_recommendations)
    else:
        if recommended_events_df.empty:
            return recommend_popular_events(df_events_future, df_users_past, n_recommendations)
        recommended_events_df = recommended_events_df.head(n_recommendations)
    
    return recommended_events_df[['event_id', 'name', 'type', 'start_datetime']]

def recommend_events_collaborative(user_id, preds_df, df_events_future, df_users_past, num_recommendations=5):
    """
    Recommend events to a user based on collaborative filtering predictions.
    """
    if user_id not in preds_df.index:
        return recommend_popular_events(df_events_future, df_users_past, num_recommendations)
    
    # Get the user's predicted ratings
    user_predictions = preds_df.loc[user_id]
    user_predictions = user_predictions[user_predictions > 0]
    user_predictions = user_predictions.sort_values(ascending=False)
    user_predictions = user_predictions.reset_index()
    user_predictions.columns = ['event_id', 'PredictedRating']
    
    # Get events the user has already rated
    user_events = set(df_users_past[df_users_past['user_id'] == user_id]['event_id'].tolist())
    
    # Filter to future events not yet rated by the user
    recommendations = df_events_future[~df_events_future['event_id'].isin(user_events)]
    
    # Merge with predicted ratings
    recommendations = recommendations.merge(user_predictions, on='event_id', how='inner')
    if recommendations.empty:
        return recommend_popular_events(df_events_future, df_users_past, num_recommendations)
    
    # Select top-N recommendations based on predicted ratings
    recommendations = recommendations.sort_values('PredictedRating', ascending=False)
    recommendations = recommendations[['event_id', 'name', 'type', 'start_datetime', 'PredictedRating']]
    
    return recommendations.head(num_recommendations)

def generate_hybrid_recommendations(user_id, preds_df, df_events_future, df_users_past, user_similarity_df, alpha=0.7, beta=0.3, num_recommendations=5):
    """
    Generate hybrid recommendations by combining CF and CB recommendations using weighted scores.
    """
    # Get CF recommendations
    cf_recs = recommend_events_collaborative(user_id, preds_df, df_events_future, df_users_past, num_recommendations * 2)
    
    # Get CB recommendations based on similar users
    cb_recs = recommend_events_based_on_similar_users(user_id, df_users_past, df_events_future, user_similarity_df, n_similar_users=5, n_recommendations=num_recommendations * 2)
    
    # Assign scores
    if not cf_recs.empty:
        cf_recs = cf_recs.copy()
        cf_recs['CF_Score'] = cf_recs['PredictedRating']
    else:
        cf_recs = pd.DataFrame(columns=['event_id', 'name', 'type', 'start_datetime', 'CF_Score'])
    
    if not cb_recs.empty:
        cb_recs = cb_recs.copy()
        cb_recs['CB_Score'] = 1  # Assign a uniform score for CB recommendations
    else:
        cb_recs = pd.DataFrame(columns=['event_id', 'name', 'type', 'start_datetime', 'CB_Score'])
    
    # Merge CF and CB recommendations
    combined_recs = pd.concat([cf_recs, cb_recs], ignore_index=True, sort=False)
    
    # Fill missing scores with zero
    combined_recs[['CF_Score', 'CB_Score']] = combined_recs[['CF_Score', 'CB_Score']].fillna(0)
    
    # Calculate TotalScore as a weighted sum
    combined_recs['TotalScore'] = alpha * combined_recs['CF_Score'] + beta * combined_recs['CB_Score']
    
    # Sort by TotalScore in descending order
    combined_recs = combined_recs.sort_values('TotalScore', ascending=False)
    
    # Remove duplicate event_id entries, keeping the one with the highest TotalScore
    combined_recs = combined_recs.drop_duplicates(subset='event_id', keep='first')
    
    # Select top-N recommendations
    recommendations = combined_recs[['event_id', 'name', 'type', 'start_datetime']].head(num_recommendations)
    
    return recommendations

# ---------------------------
# 5. Example Usage
# ---------------------------

def generate_recommendations_for_user(user_id, preds_df, df_events_future, df_users_past, user_similarity_df, num_recommendations=5):
    """
    Generate and display hybrid recommendations for a specified user.
    """
    recommendations = generate_hybrid_recommendations(
        user_id, preds_df, df_events_future, df_users_past, user_similarity_df, 
        alpha=0.7, beta=0.3, num_recommendations=num_recommendations
    )
    
    if not recommendations.empty:
        print(f"\nTop {num_recommendations} Recommendations for User {user_id}:")
        print(recommendations)
    else:
        print(f"No recommendations available for User {user_id}.")

# Replace with the desired user ID
target_user_id = 5

# Generate and display recommendations
generate_recommendations_for_user(
    target_user_id, preds_df, df_events_future, df_users_past, user_similarity_df, num_recommendations=5
)
