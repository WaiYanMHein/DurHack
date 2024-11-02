import random
import pandas as pd


# ==================================================User Data==================================================

# Generate dummy data
users = [i for i in range(1, 21)]
events = [j for j in range(1, 11)]


def generate_availability():
    # Generate random availability times between 9 and 18
    available_hours = random.sample(range(9, 19), random.randint(8, 9))
    return sorted(available_hours)


def generat_rating():
    return round(random.uniform(0, 10), 1)


user_data = []

for user in users:
    event = random.choice(events)
    rating = generat_rating()
    availability = generate_availability()
    user_data.append(
        {
            "user_id": user,
            "event_id": event,
            "rating": rating,
            "availability": availability,
        }
    )

# Convert to DataFrame for better visualization
df = pd.DataFrame(user_data)
print(df)
print("\n")


# ==================================================Generated Events==================================================

events_df = pd.read_csv("events_small copy.csv")
print(events_df)
print("\n")


# ==================================================Fitting Time==================================================
def fitting_time():

    event_time = []

    for _, row in events_df.iterrows():
        event_id = row["event_id"]
        start_time = pd.to_datetime(row["start_time"]).hour
        end_time = pd.to_datetime(row["end_time"]).hour
        duration = list(range(start_time, end_time))
        event_time.append({"event_id": event_id, "duration": duration})

    # Create a dictionary for quick lookup of event durations
    event_durations_dict = {
        event["event_id"]: event["duration"] for event in event_time
    }

    # Create a new array to store {user_id, user_availability, event_duration}
    combined_info = []

    for user in user_data:
        event_id = user["event_id"]
        user_availability = user["availability"]
        matching_events = []
        for event in event_durations_dict.items():
            if all(hour in user_availability for hour in event[1]):
                matching_events.append(event[0])

        combined_info.append(
            {
                "user_id": user["user_id"],
                # "user_availability": user_availability,
                "matching_events": matching_events,
            }
        )
    # Convert to DataFrame for better visualization
    combined_df = pd.DataFrame(combined_info)
    return combined_df


print(fitting_time())
