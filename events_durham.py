import csv
import random
from datetime import datetime, timedelta
import numpy as np

# Parameters


num_events = 1000
start_date = datetime(2024, 6, 7)
end_date = datetime(2024, 12, 13)
time_slots = [timedelta(hours=h, minutes=m) for h in range(8, 24) for m in [0, 15, 30, 45]]
event_types = ['Concert', 'Exhibition', 'Workshop', 'Networking Event', 'Fair', 'Sports Event', 'Play', 'Movie Screening', 'Seminar', 'Charity Event', 'Lecture', 'Festival', 'Ball', 'Party', 'Conference']
locations = ['Durham, United Kingdom']
event_names = ['Music Festival', 'Art Exhibition', 'Startup Workshop', 'Networking Evening', 'Food Fair', 'Sports Day', 'Shakespeare Play', 'Film Night', 'Science Seminar', 'Charity Run', 'Historical Lecture', 'Cultural Festival', 'Winter Gala', 'New Year Party', 'Tech Conference']

# Generate event data
events = []
date_range = (end_date - start_date).days + 1
event_dates = [start_date + timedelta(days=x) for x in range(date_range)]

for event_id in range(1, num_events + 1):
    # Random date
    date = random.choice(event_dates)

# Random start time (mostly in the afternoon)
    if random.random() < 0.7:
        start_time = random.choice(time_slots[20:])  # Afternoon slots
    else:
        start_time = random.choice(time_slots)

# Random duration with mean ~1.5h, max 4h
    duration_minutes = min(int(random.expovariate(1/90)), 240)  # Mean 90 minutes
    duration = timedelta(minutes=(duration_minutes // 15) * 15)  # Round to nearest 15 mins

    end_time = start_time + duration
    if end_time > timedelta(hours=24):
        end_time = timedelta(hours=23, minutes=59)

    start_datetime = datetime.combine(date, datetime.min.time()) + start_time
    end_datetime = datetime.combine(date, datetime.min.time()) + end_time

    # Random event type and name
    event_type = random.choice(event_types)
    name = f"{random.choice(event_names)} #{event_id}"

    event = {
        '': event_id,
        'event_id': event_id,
        'name': name,
        'type': event_type,
        'location': random.choice(locations),
        'start_time': start_datetime.strftime('%H:%M:%S'),
        'end_time': end_datetime.strftime('%H:%M:%S'),
        'date': date.strftime('%Y-%m-%d'),
        'start_datetime': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_datetime': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
    }
    events.append(event)

# Write to CSV
with open('events_durham.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['', 'event_id', 'name', 'type', 'location', 'start_time', 'end_time', 'date', 'start_datetime', 'end_datetime']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for event in events:
        writer.writerow(event)


################################################################
#users_durham_past 

# Parameters
num_users = 500
average_events_per_user = 30
max_rating = 5
cutoff_date = datetime(2024, 11, 2)

# Filter events before cutoff date
past_events = [event for event in events if datetime.strptime(event['date'], '%Y-%m-%d') < cutoff_date]

users_past = []

for user_id in range(1, num_users + 1):
    # Number of events attended by the user
    num_events_attended = np.random.poisson(average_events_per_user)
    num_events_attended = min(num_events_attended, len(past_events))
    
    # Randomly select events attended
    attended_events = random.sample(past_events, num_events_attended)
    
    for event in attended_events:
        # Generate rating using Poisson distribution with mean 3.5
        rating = np.random.poisson(3.5)
        
        # Cap the rating at 5
        rating = min(rating, max_rating)
        
        # Ensure rating is within 0 to 5
        rating = max(0, rating)
        
        users_past.append({
            '': len(users_past) + 1,
            'user_id': user_id,
            'event_id': event['event_id'],
            'rating': rating
        })

# Write to CSV
with open('users_past_durham.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['', 'user_id', 'event_id', 'rating']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for entry in users_past:
        writer.writerow(entry)

################################################################
#users_durham_future
# Filter events after cutoff date
future_events = [event for event in events if datetime.strptime(event['date'], '%Y-%m-%d') >= cutoff_date]

users_going = []

for user_id in range(1, num_users + 1):
    for event in future_events:
        going = random.choice([True, False])
        users_going.append({
            '': len(users_going) + 1,
            'user_id': user_id,
            'event_id': event['event_id'],
            'going': going
        })

# Write to CSV
with open('users_going_durham.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['', 'user_id', 'event_id', 'going']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for entry in users_going:
        writer.writerow(entry)

