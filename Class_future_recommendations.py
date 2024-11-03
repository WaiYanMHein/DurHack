import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


class FutureRecommendations:
    def __init__(
        self, events_past_path, ratings_past_path, events_future_path, going_future_path
    ):
        self.events_past_path = events_past_path
        self.ratings_past_path = ratings_past_path
        self.events_future_path = events_future_path
        self.going_future_path = going_future_path
        self.df_events_past = None
        self.df_ratings_past = None
        self.df_events_future = None
        self.df_going_future = None

    def load_data(self):
        self.df_events_past = pd.read_csv(self.events_past_path)
        self.df_ratings_past = pd.read_csv(self.ratings_past_path)
        self.df_events_future = pd.read_csv(self.events_future_path)
        self.df_going_future = pd.read_csv(self.going_future_path)

    def clean_data(self):
        self.df_events_past["start_datetime"] = pd.to_datetime(
            self.df_events_past["start_datetime"]
        )
        self.df_events_future["start_datetime"] = pd.to_datetime(
            self.df_events_future["start_datetime"]
        )

    def cast_booleans_to_ints(self):
        self.df_going_future["going"] = self.df_going_future["going"].astype(int)

    def merge_data(self):
        self.df_ratings_past = self.df_ratings_past.merge(
            self.df_events_past, how="left", on="event_id"
        )

    def run(self):
        self.load_data()
        self.clean_data()
        self.cast_booleans_to_ints()
        self.merge_data()
        # Add more methods or code to perform the recommendations


# Example usage
if __name__ == "__main__":
    recommender = FutureRecommendations(
        events_past_path="correct_csvs/events_past.csv",
        ratings_past_path="correct_csvs/ratings_past.csv",
        events_future_path="correct_csvs/events_future.csv",
        going_future_path="correct_csvs/going_future.csv",
    )
    recommender.run()
