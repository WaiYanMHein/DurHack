# Events recommendation system for Durham Society 

## AI recommendation system 
- Content-based 
- Item-to-item collaborative filtering
  
## Purpose 
- By using Machine learning alogorithm, it generates similar future events for users to choose if they wish to attend or not
- data will be trained after user submitting their response

## Items
- #### future_recommendations.py
  - generate 50 similar future events based on user's rating
- #### cal.py
  - it import user's calender to a list of availability time slot within user's timetable
- #### fitting_time.py
  - it filters the generated future event by checking user's availbility from cal.py
  - it shows events that users can attend within the period
- #### chainofevent.py
  - it combines all the file to output a list of {date, matching_event}
- #### events_durham.csv
  - it stores the future durham society events
- #### ratings_past.csv
  - it stores user info, including the rating of events
## Procedure
- User attended events and submit their attending history, rating to the program
- generate similar events using the attending history of users
- separating past and future events for the generated events
- check user's timetable to see if the future events can be attended by the user, eliminiate event that has time conflict with user
- input a period of time, such as "2024-11-01" to "2024-11-09", with user ID "43", check to see which events will be recommended to the user within the period of time

