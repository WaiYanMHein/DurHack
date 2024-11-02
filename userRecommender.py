from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Dummy dataset
users = [
    {"userID": "user1", "username": "John Doe"},
    {"userID": "user2", "username": "Jane Smith"},
]

events = [
    {"eventID": "event1", "name": "Event 1"},
    {"eventID": "event2", "name": "Event 2"},
    {"eventID": "event3", "name": "Event 3"},
    {"eventID": "event4", "name": "Event 4"},
]

# Dummy event history (userID, eventID, rating)
event_history = [
    ("user1", "event1", 5),
    ("user1", "event2", 3),
    ("user1", "event3", 2),
    ("user2", "event1", 2),
    ("user2", "event3", 5),
    ("user2", "event4", 3),
]

# Load the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    pd.DataFrame(event_history, columns=["userID", "eventID", "rating"]), reader
)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm for collaborative filtering
algo = SVD()
algo.fit(trainset)

# Evaluate the algorithm
predictions = algo.test(testset)
accuracy.rmse(predictions)


def recommend_events(user_id, top_n=3):
    user_events = [event for event in event_history if event[0] == user_id]
    user_event_ids = [event[1] for event in user_events]

    # Predict ratings for all events the user hasn't rated yet
    all_event_ids = [event["eventID"] for event in events]
    unrated_event_ids = [
        event_id for event_id in all_event_ids if event_id not in user_event_ids
    ]

    predictions = [algo.predict(user_id, event_id) for event_id in unrated_event_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top N recommendations
    top_recommendations = predictions[:top_n]
    recommended_events = [get_event_data(pred.iid) for pred in top_recommendations]

    return recommended_events


def get_event_data(event_id):
    for event in events:
        if event["eventID"] == event_id:
            return event
    return None


if __name__ == "__main__":
    user_id = "user1"
    recommendations = recommend_events(user_id)
    for event in recommendations:
        print(f"Recommended Event: {event['eventID']} - {event['name']}")
