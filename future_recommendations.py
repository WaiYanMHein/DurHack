import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


## Read in dataset
df_events_past = pd.read_csv("correct_csvs/events_past.csv")
df_ratings_past = pd.read_csv("correct_csvs/ratings_past.csv")
df_events_future = pd.read_csv("correct_csvs/events_future.csv")
df_going_future = pd.read_csv("correct_csvs/going_future.csv")


## Only use the start_date_times
## Clean up the datetimes
df_events_past["start_datetime"] = pd.to_datetime(df_events_past["start_datetime"])
df_events_future["end_datetime"] = pd.to_datetime(df_events_future["start_datetime"])

## Cast booleans to ints for kNN
df_going_future["going"] = df_going_future["going"].astype(int)


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

    user_index = [user_mapper[i] for i in df["user_id"]]
    item_index = [event_mapper[i] for i in df["event_id"]]

    X = csr_matrix(
        (df[column], (user_index, item_index)), shape=(numUniqueUsers, numUniqueevents)
    )

    return X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper


def train_model(X, k, metric="cosine"):
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
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
        vector = vector.reshape(1, -1)

    neighbour = kNN.kneighbors(vector, return_distance=False)
    for i in range(1, k):
        n = neighbour.item(i)
        neighbourIDs.append(inv_mapper[n])
    return neighbourIDs


def get_similar_users(user_id, k):
    """
    Get most similar users as a dataframe

    Args:
        user_id: the user id of the event the user liked

    Returns: user IDs of similar users"""

    similar_users = find_kNN(
        user_id, trained_kNN_past_data, pastX, user_mapper, user_inv_mapper, k
    )
    return similar_users


def get_similar_events_future(event_id, k):
    """
    Get most similar users as a list of user ids

    Args:
        user_id: the user id of the event the user liked

    Returns: user IDs of similar users"""

    similar_users = find_kNN(
        event_id, trained_kNN_future_data, futureX, event_mapper, event_inv_mapper, k
    )
    return similar_users


def get_going(user_ids):
    """Takes a list of users and finds the events they are going to
    Args:
    user_ids: a list of user ids

    Returns: list of all the events these users are going to"""
    df_going = df_going_future[df_going_future["user_id"].isin(user_ids)]
    going_event_ids = df_going["event_id"].unique().tolist()
    return going_event_ids


def recommend_events(user_id):
    """
    Take a user id and find event recommendations in the future

    Returns: Event ids for recommended events"""

    similar_users = get_similar_users(user_id, kPast)
    recommend_events = get_going(similar_users)

    return recommend_events


def user_choose_time(start_time, end_time):
    """
    Prompt the user to choose a date range for event recommendations.

    Args:
        start_time: the start datetime for the range
        end_time: the end datetime for the range

    Returns: DataFrame of events within the specified date range
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    filtered_events = df_events_future[
        (df_events_future["start_datetime"] >= start_time)
        & (df_events_future["start_datetime"] <= end_time)
    ]

    return filtered_events


def closest_time_events(recommended_events, num):
    """
    find the n events that are closest to current date
    Args:
        recommended_events (_type_): _description_

    """
    recommended_events_df = df_events_future[
        df_events_future["event_id"].isin(recommended_events)
    ]
    recommended_events_df = recommended_events_df.sort_values(by="start_datetime")
    return recommended_events_df.head(num)


## Train first tower --- finding similar users
pastX, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper = create_matrix(
    df_ratings_past, "rating"
)
kPast = 5
trained_kNN_past_data = train_model(pastX, kPast)


def generate_recommendations(user_id):
    """
    Generate recommendations for a user

    Args:
        user_id: the user id of the event the user liked

    Returns: list of event ids for events which are similar
    """
    generated_recommended_events = recommend_events(user_id)
    time_reco_events = closest_time_events(generated_recommended_events, 50)
    return time_reco_events


# print(generate_recommendations(1))
## Train first tower --- finding similar users
# pastX, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper = create_matrix(
#     df_ratings_past, "rating"
# )
# kPast = 5
# trained_kNN_past_data = train_model(pastX, kPast)


## Not used------------
## Train second tower --- finding similar future events
futureX, user_mapper, event_mapper, user_inv_mapper_inv_mapper, event_inv_mapper = (
    create_matrix(df_going_future, "going")
)
kFuture = 10
futureX = futureX.T  ##Transpose since grouping similar events
trained_kNN_future_data = train_model(futureX, kFuture)
##----------------------------

# Example usage
# print("Time range for events: ")
# start_time = input("Enter the start time: ")
# end_time = input("Enter the end time: ")
# filtered_events = user_choose_time(start_time, end_time)
# user_id = 5  # Example user_id

# print(f"Recommended events for user {user_id}: {recommended_events}")
# print(f"Recommended events for user {user_id}: {time_reco_events}")
