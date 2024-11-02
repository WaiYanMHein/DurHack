import pandas as pd


def read_events(filename='/Users/morgan/PycharmProjects/Durhack/venv/events_10_two_weeks.csv'):
    """Read events from a CSV file."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError as e:
        print(f"Error: The file '{filename}' was not found.")
        raise e
    except pd.errors.EmptyDataError as e:
        print(f"Error: The file '{filename}' is empty.")
        raise e
    except pd.errors.ParserError as e:
        print(f"Error: The file '{filename}' is not in valid CSV format.")
        raise e


def prompt_user_for_events(events):
    """Prompt the user to select events they want to attend."""
    interested_events = []

    for index, event in events.iterrows():
        print(f"Event ID: {event['event_id']}, Event Name: {event['name']}, Date: {event['date']}")
        response = input("Do you want to attend this event? (yes/no): ").strip().lower()
        if response == 'yes':
            interested_events.append(event['event_id'])

    return interested_events


def update_future_csv(user_id, interested_events, future_file='/Users/morgan/PycharmProjects/Durhack/venv/cal/future.csv'):
    """
    Update the future.csv file with the user's event attendance preferences.

    :param user_id: ID of the user
    :param interested_events: List of event IDs the user wants to attend
    :param future_file: The future CSV file to be updated
    """
    try:
        # Read the existing future.csv file
        future_data = pd.read_csv(future_file)
    except FileNotFoundError:
        # If the CSV file does not exist, create an empty DataFrame with the required structure
        future_data = pd.DataFrame(columns=['user_id', 'event_id'])

    for event_id in interested_events:
        future_data = future_data.append({'user_id': user_id, 'event_id': event_id}, ignore_index=True)

    # Write the updated data back to the future.csv file
    future_data.to_csv(future_file, index=False)


# Example usage
if __name__ == "__main__":
    events_df = read_events()
    user_id = input("Enter your user ID: ").strip()
    interested_events_ids = prompt_user_for_events(events_df)
    update_future_csv(user_id, interested_events_ids)