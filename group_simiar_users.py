import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix    
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


## Read in dataset
df_events_past = pd.read_csv("past_future_data/events_filtered_past.csv")
df_ratings_past = pd.read_csv("past_future_data/ratings_filtered_past.csv")
df_events_future = pd.read_csv("past_future_data/events_filtered_future.csv")
df_going_future = pd.read_csv("past_future_data/going_filtered_future.csv")



## Only use the start_date_times
## Clean up the datetimes
df_events_past["start_datetime"] = pd.to_datetime(df_events_past["start_datetime"])
df_events_future["end_datetime"] = pd.to_datetime(df_events_future["start_datetime"])

## Cast booleans to ints for kNN
df_going_future['going'] = df_going_future['going'].astype(int)



## Merge the data on movies into the ratings dataframe
df_ratings_past = df_ratings_past.merge(df_events_past, how="left", on="event_id")




def create_matrix(df, column):
    """
    Creates a sparse matrix from the past ratings dataframe.

    Args:
        df: pandas dataframe with 3 columns: (user_id, event_id, rating)
        column: the column with the data to group by
      
    Returns:
        X: sparse matrix
        user_mapper: dict mapping user_id to user indicies
        user_inv_mapper: dict that maps user indices to user id's
        event_mapper: dict that maps event id's to event indices
        event_inv_mapper: dict that maps event indices to event id's
    """

    numUniqueUsers = df["user_id"].nunique()
    numUniqueevents = df["event_id"].nunique()

    user_ids = np.unique(df["user_id"])
    userIndecies = list(range(numUniqueUsers))
    eventIds = np.unique(df["event_id"])
    eventIndecies = list(range(numUniqueevents))

    user_mapper = dict(zip(user_ids, userIndecies))
    user_inv_mapper = dict(zip(userIndecies, user_ids))
    event_mapper = dict(zip(eventIds, eventIndecies))
    event_inv_mapper = dict(zip(eventIndecies, eventIds))

    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [event_mapper[i] for i in df['event_id']]

    X = csr_matrix((df[column], (user_index,item_index)), shape=(numUniqueUsers, numUniqueevents))
    
    return X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper
    



def train_model(X, k, metric="cosine"):
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    return kNN



def find_kNN(id, kNN, X, mapper, inv_mapper, k):
    """
    Find k-nearest neighbours
    
    Args:
        user_id: id
        kNN: the fitted model
        X: user-item utility matrix
        metric: distance metric for kNN calculations

    Returns: list of k most similar
    """

    neighbourIDs = []
    index = mapper[id]

    vector = X[index]

    if isinstance(vector, (np.ndarray)):
        vector = vector.reshape(1,-1)

    neighbour = kNN.kneighbors(vector, return_distance=False)
    for i in range(1,k):
        n = neighbour.item(i)
        neighbourIDs.append(inv_mapper[n])
    return neighbourIDs



def get_similar_users(user_id, k):
    """
    Get most similar users as a dataframe
    
    Args:
        user_id: the user id of the event the user liked
        
    Returns: user IDs of similar users"""

    similar_users = find_kNN(user_id, trained_kNN_past_data, pastX, user_mapper, user_inv_mapper, k)
    return similar_users


def get_similar_events_future(event_id, k):
    """
    Get most similar users as a dataframe
    
    Args:
        user_id: the user id of the event the user liked
        
    Returns: user IDs of similar users"""

    similar_users = find_kNN(event_id, trained_kNN_future_data, futureX, event_mapper, event_inv_mapper, k)
    return similar_users



def recommend_events(user_id):
    """
    Take a user id and find event recommendations in the future
    
    Returns: Event ids"""

    similar_users = get_similar_users(user_id)
    recommendation_ids = []

    for similar_user_id in similar_users:
        get_similar_events_future




event_id_to_name = dict(zip(df_events_past["event_id"], df_events_past["name"]))
event_name_to_id = dict(zip(df_events_past["name"], df_events_past["event_id"]))

## Train first tower --- finding similar users
pastX, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper = create_matrix(df_ratings_past, "rating")
kPast = 10
trained_kNN_past_data = train_model(pastX, kPast)

## Train second tower --- finding similar future events
futureX, user_mapper, event_mapper, user_inv_mapper_inv_mapper, event_inv_mapper = create_matrix(df_going_future, "going")
kFuture = 10
futureX = futureX.T ##Transpose since grouping similar events
trained_kNN_future_data = train_model(futureX, kFuture)






