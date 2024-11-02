from icalendar import Calendar
from datetime import datetime, timedelta, time

# Replace 'my_calendar.ics' with the path to your .ics file
ICS_FILE_PATH = ']/Users/morgan/Documents/GitHub/DurHack/morgan22068@gmail.com.ics'

# Define the date range for which you want to find free slots
start_date = datetime.now().date()  # Start date (today)
end_date = start_date + timedelta(days=7)  # End date (7 days from today)

# Define working hours (e.g., 9 AM to 5 PM)
WORKING_HOURS_START = time(9, 0)  # 9:00 AM
WORKING_HOURS_END = time(17, 0)   # 5:00 PM

def load_events(ics_file_path):
    with open(ics_file_path, 'r') as f:
        calendar = Calendar.from_ical(f.read())

    events = []
    for component in calendar.walk():
        if component.name == "VEVENT":
            dtstart = component.get('dtstart').dt
            dtend = component.get('dtend').dt

            # Handle all-day events
            if isinstance(dtstart, datetime) and isinstance(dtend, datetime):
                events.append((dtstart, dtend))
            elif isinstance(dtstart, datetime.date) and isinstance(dtend, datetime.date):
                # All-day event spans multiple days
                current_date = dtstart
                while current_date <= dtend:
                    events.append((
                        datetime.combine(current_date, WORKING_HOURS_START),
                        datetime.combine(current_date, WORKING_HOURS_END)
                    ))
                    current_date += timedelta(days=1)
            else:
                # Mixed datetime and date types
                if isinstance(dtstart, datetime.date):
                    dtstart = datetime.combine(dtstart, WORKING_HOURS_START)
                if isinstance(dtend, datetime.date):
                    dtend = datetime.combine(dtend, WORKING_HOURS_END)
                events.append((dtstart, dtend))
    return events

def find_free_slots(events, date_to_check):
    # Initialize the start and end of the working day
    working_day_start = datetime.combine(date_to_check, WORKING_HOURS_START)
    working_day_end = datetime.combine(date_to_check, WORKING_HOURS_END)

    # Filter events for the specific date
    daily_events = []
    for event_start, event_end in events:
        # If the event occurs on the date_to_check
        if event_start.date() <= date_to_check <= event_end.date():
            # Adjust the event times to the working day boundaries
            event_start = max(event_start, working_day_start)
            event_end = min(event_end, working_day_end)
            if event_start < event_end:
                daily_events.append((event_start, event_end))

    # Sort events by start time
    daily_events.sort(key=lambda x: x[0])

    free_slots = []
    current_time = working_day_start

    for event_start, event_end in daily_events:
        if current_time >= working_day_end:
            break

        if event_start > current_time:
            free_slots.append((current_time, event_start))

        current_time = max(current_time, event_end)

    # Check for free time after the last event
    if current_time < working_day_end:
        free_slots.append((current_time, working_day_end))

    return free_slots

def print_free_slots(date, free_slots):
    if not free_slots:
        print(f"No free slots available on {date.strftime('%Y-%m-%d')}.")
        return

    print(f"Free time slots on {date.strftime('%Y-%m-%d')}:")
    for start, end in free_slots:
        duration = end - start
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        print(f"  {start.strftime('%H:%M')} - {end.strftime('%H:%M')} ({hours}h {minutes}m)")
    print()

def main():
    # Load all events once
    events = load_events(ICS_FILE_PATH)

    # Loop over each date in the date range
    current_date = start_date
    while current_date <= end_date:
        free_slots = find_free_slots(events, current_date)
        print_free_slots(current_date, free_slots)
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
