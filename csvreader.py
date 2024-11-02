import csv
import pandas as pd

def read_csv_file(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        data = [row for row in csv_reader]
    return header, data

def csv_to_dataframe(file_path):
    header, data = read_csv_file(file_path)
    df = pd.DataFrame(data, columns=header)
    return df

df = csv_to_dataframe('users_updated.csv')
print(df)
#print(read_csv_file('users_updated.csv'))