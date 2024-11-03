import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse.linalg import svds


# Load datasets
df_events = pd.read_csv("events_durham.csv")
df_users_past = pd.read_csv("users_past_durham.csv")
df_users_going = pd.read_csv("users_going_durham.csv")

# Convert date columns to datetime
df_events['start_datetime'] = pd.to_datetime(df_events['start_datetime'], format='%d/%m/%Y %H:%M', errors='coerce')
df_events['end_datetime'] = pd.to_datetime(df_events['end_datetime'], format='%d/%m/%Y %H:%M', errors='coerce')

# Define cutoff date
cutoff_date = datetime(2024, 11, 2)

# Split events
df_events_past = df_events[df_events['start_datetime'] < cutoff_date].copy()
df_events_future = df_events[df_events['start_datetime'] >= cutoff_date].copy()

# Filter user data
df_users_past = df_users_past[df_users_past['event_id'].isin(df_events_past['event_id'])]
df_users_going['going'] = df_users_going['going'].astype(int)

# Ensure user_id and event_id are integers
df_users_past['user_id'] = df_users_past['user_id'].astype(int)
df_users_past['event_id'] = df_users_past['event_id'].astype(int)
df_users_past['rating'] = df_users_past['rating'].astype(float)

# Prepare combined features
df_events['combined_features'] = df_events['type'] + ' ' + df_events['name']

# Vectorize features for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_events['combined_features'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create mappings
event_id_to_index = pd.Series(df_events.index, index=df_events['event_id']).drop_duplicates()
index_to_event_id = pd.Series(df_events['event_id'], index=df_events.index)

# Functions for content-based filtering
def get_similar_events(event_id, n_similar=10):
    if event_id not in event_id_to_index:
        return []
    idx = event_id_to_index[event_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_similar+1]
    event_indices = [i[0] for i in sim_scores]
    return index_to_event_id.iloc[event_indices].tolist()

def get_user_liked_events(user_id, min_rating=4):
    liked_events = df_users_past[
        (df_users_past['user_id'] == user_id) & (df_users_past['rating'] >= min_rating)
    ]['event_id'].tolist()
    return liked_events

def recommend_events_for_user(user_id, n_recommendations=5):
    liked_events = get_user_liked_events(user_id)
    if not liked_events:
        # Return popular future events as fallback
        popular_events = df_events_future['event_id'].value_counts().index.tolist()
        recommended_events = df_events_future[df_events_future['event_id'].isin(popular_events)]
        return recommended_events.head(n_recommendations)
    similar_events = []
    for event_id in liked_events:
        similar_events.extend(get_similar_events(event_id, n_similar=10))
    similar_events = list(set(similar_events) - set(liked_events))
    future_event_ids = df_events_future['event_id'].tolist()
    recommended_event_ids = [eid for eid in similar_events if eid in future_event_ids]
    recommended_events = df_events_future[df_events_future['event_id'].isin(recommended_event_ids)]
    return recommended_events.head(n_recommendations)

# Collaborative Filtering implementation
# Create user-item interaction matrix
user_item_matrix = df_users_past.pivot_table(
    index='user_id',
    columns='event_id',
    values='rating'
).fillna(0)

# Normalize the user-item matrix
user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
user_item_matrix_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

# Perform SVD
U, sigma, Vt = svds(user_item_matrix_demeaned, k=50)
sigma = np.diag(sigma)

# Reconstruct the user-item matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Function for collaborative filtering recommendations
def recommend_events_collaborative(user_id, preds_df, df_events_future, df_users_past, num_recommendations=5):
    if user_id not in preds_df.index:
        print(f"User {user_id} not found in predictions.")
        return pd.DataFrame()
    # Get the user's predicted ratings
    user_predictions = preds_df.loc[user_id].sort_values(ascending=False)
    user_predictions = user_predictions.reset_index()
    user_predictions.columns = ['event_id', 'PredictedRating']

    # Get user's historical data
    user_history = df_users_past[df_users_past.user_id == user_id]
    user_events = user_history['event_id'].tolist()

    # Exclude events the user has already rated
    recommendations = df_events_future[~df_events_future['event_id'].isin(user_events)]

    # Merge with predicted ratings
    recommendations = recommendations.merge(user_predictions, on='event_id')

    # Select top-N recommendations
    recommendations = recommendations.sort_values('PredictedRating', ascending=False)
    recommendations = recommendations[['event_id', 'name', 'type', 'start_datetime', 'PredictedRating']]
    return recommendations.head(num_recommendations)

# Hybrid recommendation function
def hybrid_recommendations(user_id, num_recommendations=5):
    # Get recommendations
    cf_recs = recommend_events_collaborative(
        user_id, preds_df, df_events_future, df_users_past, num_recommendations * 2
    )
    content_recs = recommend_events_for_user(user_id, n_recommendations=num_recommendations * 2)

    # If both methods return empty, provide popular events as fallback
    if cf_recs.empty and content_recs.empty:
        popular_events = df_events_future['event_id'].value_counts().index.tolist()
        recommendations = df_events_future[df_events_future['event_id'].isin(popular_events)]
        return recommendations.head(num_recommendations)

    # Add a 'CB_Score' column to content-based recommendations
    content_recs = content_recs.copy()
    content_recs['CB_Score'] = 1
    content_recs['CF_Score'] = 0  # Since CF score is not available here

    # Add a 'CF_Score' column to collaborative recommendations
    cf_recs = cf_recs.copy()
    cf_recs['CF_Score'] = cf_recs['PredictedRating']
    cf_recs['CB_Score'] = 0  # Since CB score is not available here

    # Combine recommendations
    combined_recs = pd.concat([cf_recs, content_recs], ignore_index=True, sort=False)

    # Remove duplicates
    combined_recs = combined_recs.drop_duplicates(subset='event_id')

    # Fill missing scores with zero
    combined_recs['CF_Score'] = combined_recs['CF_Score'].fillna(0)
    combined_recs['CB_Score'] = combined_recs['CB_Score'].fillna(0)

    # Adjust weights as needed
    alpha = 0.7  # Weight for collaborative filtering
    beta = 0.3   # Weight for content-based filtering
    combined_recs['TotalScore'] = alpha * combined_recs['CF_Score'] + beta * combined_recs['CB_Score']

    # Sort by total score
    combined_recs = combined_recs.sort_values('TotalScore', ascending=False)

    # Return top N recommendations
    return combined_recs[['event_id', 'name', 'type', 'start_datetime']].head(num_recommendations)

# Example usage
user_id = 5  # Replace with your user ID
recommendations = hybrid_recommendations(user_id, num_recommendations=5)

if not recommendations.empty:
    print(f"\nHybrid Recommendations for user {user_id}:")
    print(recommendations)
else:
    print(f"No recommendations available for user {user_id}.")

#------------------------------------------------------------
#testing


