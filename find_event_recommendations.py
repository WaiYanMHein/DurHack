import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix    
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


## Read in dataset
df_events = pd.read_csv("events_updated.csv")
df_ratings = pd.read_csv("users_updated.csv")

## Clean up the dates
df_events["date"] = pd.to_datetime(df_events["date"])
df_events['start_time'] = pd.to_datetime(df_events['start_time'], format='%H:%M:%S').dt.time
df_events['end_time'] = pd.to_datetime(df_events['end_time'], format='%H:%M:%S').dt.time

df_events['start_datetime'] = df_events.apply(
    lambda row: datetime.combine(row['date'], row['start_time']), axis=1
)

df_events['end_datetime'] = df_events.apply(
    lambda row: datetime.combine(row['date'], row['end_time']), axis=1
)


## Merge the data on movies into the ratings dataframe
df_ratings = df_ratings.merge(df_events, how="left", on="event_id")



## Count and mean of ratings for each event
df_event_stats = df_ratings.groupby("event_id")["rating"].agg(["count", "mean"])

## For calculating the bayesian avg
event_mean_counts = df_event_stats["count"].mean()
event_mean_mean_rating = df_event_stats["mean"].mean()

def bayesian_avg(ratings):
    bayesian_avg = (event_mean_counts*event_mean_mean_rating+ratings.sum())/(event_mean_counts+ratings.count())
    return bayesian_avg

bayesian_avgs = df_ratings.groupby("event_id")["rating"].agg(bayesian_avg).sort_values(ascending=False).reset_index()
bayesian_avgs.columns = ["event_id", "bayesian_avg"]


## Merge back with the stats
df_event_stats = df_event_stats.merge(bayesian_avgs, on="event_id")



## Create sparse matrix

def create_Matrix(df):
    """
    Creates a sparse matrix from the ratings dataframe.

    Args:
        df: pandas dataframe with 3 columns: (user_id, event_id, rating)
      
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

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(numUniqueUsers, numUniqueevents))
    
    return X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper
    


def find_similar_events(event_id, X, event_mapper, event_inv_mapper, k, metric="cosine"):
    """
    Find k-nearest neighbours of the event
    
    Args:
        event_id: id of the event of interest
        X: user-item utility matrix
        k: number of similar events to retrieve
        metric: distance metric for kNN calculations

    Returns: list of k most similar event IDs'
    """

    neighbourIDs = []
    event_index = event_mapper[event_id]

    X = X.T
    event_id = X[event_index]
    if isinstance(event_id, (np.ndarray)):
        event_id = event_id.reshape(1,-1)
    
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)

    neighbour = kNN.kneighbors(event_id, return_distance=False)
    for i in range(1,k):
        n = neighbour.item(i)
        neighbourIDs.append(event_inv_mapper[n])
    return neighbourIDs





def filter_future_events(df):
    """We only want to recommend events in the future. Filter a dataframe for events in the future.
    Args:
        df: dataframe for events
        
    Returns: A modified version of the dataframe with dates from the past removed
    """

    current_datetime = datetime.now()
    # Format to match 'start_datetime' column
    current_datetime_formatted = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    df_future = df[df["start_datetime"] > current_datetime_formatted]
    return df_future




# df_ratings_future = 
event_id_to_name = dict(zip(df_events["event_id"], df_events["name"]))
event_name_to_id = dict(zip(df_events["name"], df_events["event_id"]))
X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper = create_Matrix(df_ratings)





def get_recommendations(event_id):
    """
    Find 10 recommendations for events based on liking 1 event
    
    Args:
        event_id: the event id of the event the user liked
        
    Returns: list of event_ids for events which are similar"""
    similar_events = find_similar_events(event_id, X, event_mapper, event_inv_mapper, k=30)
    return similar_events




df_similar_events = df_events[df_events["event_id"].isin(get_recommendations(1))]