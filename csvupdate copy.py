import csv
from collections import defaultdict
import datetime


def update_csv(csv_file, user_id, event_id):
    """
    Update the CSV file with a new user-event pair.

    :param csv_file: The CSV file to be updated
    :param user_id: ID of the user
    :param event_id: ID of the event
    """
    try:
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
    except FileNotFoundError:
        raise Exception(f"File {csv_file} not found.")

    for row in data:
        if row[0] == user_id and row[1] == event_id:
            row[2] = 'True'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def get_user_events(csv_file, user_id):
    """
    Get a list of events that a user has agreed to attend.

    :param csv_file: The CSV file to read from
    :param user_id: ID of the user
    :return: List of event IDs
    """
    user_events = []
    try:
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == user_id and row[2] == 'True':
                    user_events.append(row[1])
    except FileNotFoundError:
        raise Exception(f"File {csv_file} not found.")

    return user_events


def create_user_calendar(csv_file, user_id):
    """
    Create a calendar for a user based on the events they agreed to attend.

    :param csv_file: The CSV file to read from
    :param user_id: ID of the user
    """
    user_events = get_user_events(csv_file, user_id)

    # Example calendar: events with their corresponding dates
    mock_event_dates = {
        'E1': '2023-12-01',
        'E2': '2023-12-05',
        'E3': '2023-12-10'
    }

    calendar = defaultdict(list)
    for event_id in user_events:
        if event_id in mock_event_dates:
            event_date = mock_event_dates[event_id]
            calendar[event_date].append(event_id)

    print(f"Calendar for user {user_id}:")
    for date, events in calendar.items():
        print(f"{date}: {', '.join(events)}")


# Example usage
csv_file = 'events.csv'
user_id = 'U1'
event_id = 'E1'

# Update CSV file to show the user agreed to attend the event
update_csv(csv_file, user_id, event_id)

# Create and print the user's calendar
create_user_calendar(csv_file, user_id)
