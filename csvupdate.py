import csv
import pandas as pd

def update_csv(csv_file, user_id, event_id):
    """
    Update the CSV file with a new user-event pair.

    :param csv_file: The CSV file to be updated
    :param user_id: ID of the user
    :param event_id: ID of the event
    """
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    for row in data:
        if row[0] == user_id and row[1] == event_id:
            row[2] = 'True'
            
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
            
#update_csv('users_future-2.csv', '1', '2')  # Update the CSV file with user ID 1 attending event ID 2
    