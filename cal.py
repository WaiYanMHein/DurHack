import requests
from icalendar import Calendar
from datetime import datetime, timedelta, time, date

# Replace the URL with your .ics file URL
ICS_URL = "https://mytimetable.durham.ac.uk/calendar/export/386cdf20245dc0ca4da8bec78fa89d7fdf0ad6d9.ics"

# Define the date range for which you want to find free slots
start_date = datetime(2024, 11, 2).date()  # Start date
end_date = datetime(2024, 11, 5).date()  # End date

# Define working hours (e.g., 9 AM to 5 PM)
WORKING_HOURS_START = time(9, 0)  # 9:00 AM
WORKING_HOURS_END = time(22, 0)  # 10:00 PM


def load_events(ics_url):
    response = requests.get(ics_url)
    response.raise_for_status()  # Raise an error for bad responses
    calendar = Calendar.from_ical(response.text)

    events = []
    for component in calendar.walk():
        if component.name == "VEVENT":
            dtstart = component.get("dtstart").dt
            dtend = component.get("dtend").dt

            # Normalize to naive datetime (remove timezone info)
            if isinstance(dtstart, datetime):
                dtstart = dtstart.replace(tzinfo=None)
            if isinstance(dtend, datetime):
                dtend = dtend.replace(tzinfo=None)

            # Handle all-day events
            if isinstance(dtstart, datetime) and isinstance(dtend, datetime):
                events.append((dtstart, dtend))
            elif isinstance(dtstart, date) and isinstance(dtend, date):
                # All-day event spans multiple days
                current_date = dtstart
                while current_date <= dtend:
                    events.append(
                        (
                            datetime.combine(current_date, WORKING_HOURS_START),
                            datetime.combine(current_date, WORKING_HOURS_END),
                        )
                    )
                    current_date += timedelta(days=1)
            else:
                # Mixed datetime and date types
                if isinstance(dtstart, date):
                    dtstart = datetime.combine(dtstart, WORKING_HOURS_START)
                if isinstance(dtend, date):
                    dtend = datetime.combine(dtend, WORKING_HOURS_END)
                events.append((dtstart, dtend))
    return events


def find_free_slots(events, date_to_check):
    # Initialize the start and end of the working day
    working_day_start = datetime.combine(date_to_check, WORKING_HOURS_START)
    working_day_start = working_day_start.replace(
        tzinfo=None
    )  # Ensure datetime is naive
    working_day_end = datetime.combine(date_to_check, WORKING_HOURS_END)
    working_day_end = working_day_end.replace(tzinfo=None)  # Ensure datetime is naive

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
    """
    Print the available free time slots for a given date.
    Args:
        date (datetime.date): The date for which to print free time slots.
        free_slots (list of tuple): A list of tuples where each tuple contains two datetime.datetime objects
                                    representing the start and end times of a free slot.
    Returns:
        None
    Example:
        from datetime import datetime, timedelta
        date = datetime(2023, 10, 1)
        free_slots = [(datetime(2023, 10, 1, 9, 0), datetime(2023, 10, 1, 10, 0)),
                          (datetime(2023, 10, 1, 14, 0), datetime(2023, 10, 1, 15, 30))]
        print_free_slots(date, free_slots)
        Free time slots on 2023-10-01:
          09:00 - 10:00 (1h 0m)
          14:00 - 15:30 (1h 30m)
    """

    daily_data = []
    for start, end in free_slots:
        duration = end - start
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        daily_data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "availability": list(range(start.hour, end.hour)),
            }
        )
    print(daily_data)
    # if not free_slots:
    #     print(f"No free slots available on {date.strftime('%Y-%m-%d')}.")
    #     return

    # print(f"Free time slots on {date.strftime('%Y-%m-%d')}:")
    # for start, end in free_slots:
    #     duration = end - start
    #     hours, remainder = divmod(duration.seconds, 3600)
    #     minutes, _ = divmod(remainder, 60)
    #     print(
    #         f"  {start.strftime('%H:%M')} - {end.strftime('%H:%M')} ({hours}h {minutes}m)"
    #     )
    # print()


def Gain_data():
    # Load all events once
    events = load_events(ICS_URL)

    # Initialize an empty list to store the combined data
    combined_data = []

    # Loop over each date in the date range
    current_date = start_date
    while current_date <= end_date:
        free_slots = find_free_slots(events, current_date)

        for start, end in free_slots:
            start_hour = start.hour
            end_hour = end.hour
            hours_range = list(range(start_hour, end_hour + 1))

            # Check if the date already exists in combined_data
            date_str = current_date.strftime("%Y-%m-%d")
            date_entry = next(
                (item for item in combined_data if item["date"] == date_str), None
            )

            if date_entry:
                # Append the new hours to the existing availability array
                date_entry["availability"].extend(hours_range)
                # Remove duplicates and sort the array
                date_entry["availability"] = sorted(set(date_entry["availability"]))
            else:
                # Create a new entry for the date
                combined_data.append({"date": date_str, "availability": hours_range})

        current_date += timedelta(days=1)

    return combined_data


if __name__ == "__main__":
    Gain_data()
