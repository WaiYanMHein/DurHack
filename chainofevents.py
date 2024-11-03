from future_recommendations import generate_recommendations
from fitting_time import fitting_time
import pandas as pd


def main(start, end, user_id):
    Closest_event = generate_recommendations(user_id)

    # Convert Closest_event to a DataFrame
    df_closest_event = pd.DataFrame(Closest_event)
    print(df_closest_event)

    # Save the DataFrame to a CSV file
    df_closest_event.to_csv("MergeAll.csv", index=False)
    print("Closest event data has been saved to closest_event.csv")

    # use fitting_time function and MergeAll.csv
    list = fitting_time("MergeAll.csv", start, end)
    return list

def recommendations_to_csv(user_id):
    # convert the recommendations from generate_recommendations to a csv file
    Closest_event = generate_recommendations(user_id)
    df_closest_event = pd.DataFrame(Closest_event)
    df_closest_event.to_csv("MergeAll.csv", index=False)
    return "Closest event data has been saved to closest_event.csv"


if __name__ == "__main__":
    start = input("Enter the start time: ")
    end = input("Enter the end time: ")
    # in the format of "2024-11-01"
    user_id_input = int(input("Enter the user id: "))
    print(main(start, end, user_id=user_id_input))
