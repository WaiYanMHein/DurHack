import random
import pandas as pd
from cal import Gain_data


# ==================================================User Data==================================================

# Generate from the cal.py

#users = Gain_data()


# Convert to DataFrame for better visualization
#df = pd.DataFrame(users)
#print(df)
#print("\n")


# ==================================================Generated Events==================================================

"""events_df = pd.read_csv("events_future-2.csv")
print("Events", events_df)
print("\n")"""


# ==================================================Fitting Time==================================================
def fitting_time(csv_file, start, end):
    users = Gain_data(start, end)
    df = pd.DataFrame(users)
    #print(df)
    try: 
        # Read the CSV file into a DataFrame
        events_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        # Return an error message if the file is not found
        return ("File not found error")
    
    event_time = []

    # Iterate over each row in the events DataFrame
    for _, row in events_df.iterrows():
        event_id = row["event_id"]
        event_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        start_time = pd.to_datetime(row["start_time"]).hour
        end_time = pd.to_datetime(row["end_time"]).hour
        duration = list(range(start_time, end_time))
        event_time.append(
            {"event_id": event_id, "duration": duration, "date": event_date}
        )

    # Create a dictionary for quick lookup of event durations
    event_durations_dict = {
        event["event_id"]: {"date": event["date"], "duration": event["duration"]}
        for event in event_time
    }

    combined_info = []

    # Iterate over each user
    for user in users:
        date = user["date"]
        user_availability = user["availability"]
        matching_events = []
        # Check for matching events based on date and availability
        for event in event_durations_dict.items():
            if event[1]["date"] == date and all(
                hour in user_availability for hour in event[1]["duration"]
            ):
                matching_events.append(event[0])

        combined_info.append(
            {
                "date": date,
                "user_availability": user_availability,
                "matching_events": matching_events,
            }
        )
    
    # Convert the combined information to a DataFrame for better visualization
    combined_df = pd.DataFrame(combined_info)
    return combined_df


print(fitting_time("events_future-2.csv", "2024-11-02", "2024-11-06"))
