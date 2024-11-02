# Events recommendation system for Durham Society 

## AI recommendation system 
- Content-based 
- Item-to-item collaborative filtering
  
## Elements
- Dataset for events
  - Rating, type of events, etc
- Dataset for users
  - Matching similar profiles 
- Database for storing data
  - MongoDB 
- Implementing the AI system 
  - Python 
## Procedure
- User attended events and submit their attending history to the program
- generate similar events using the attending history of users
- separating past and future events for the generated events
- check user's timetable to see if the future events can be attended by the user, eliminiate event that has time conflict with user

