# DO NOT RUN 

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the datasets
events_df = pd.read_csv('events_updated.csv')
users_df = pd.read_csv('users_updated.csv')

# Display the first few rows of each dataframe
print("Events DataFrame:")
print(events_df.head())

print("\nUsers DataFrame:")
print(users_df.head())

# Handle missing values
events_df.fillna(method='ffill', inplace=True)
users_df.fillna(method='ffill', inplace=True)

# Convert categorical columns to numerical using one-hot encoding
events_df = pd.get_dummies(events_df, drop_first=True)
users_df = pd.get_dummies(users_df, drop_first=True)

# Normalize numerical columns

scaler = StandardScaler()
events_df[events_df.columns] = scaler.fit_transform(events_df[events_df.columns])
users_df[users_df.columns] = scaler.fit_transform(users_df[users_df.columns])

# Save the preprocessed data to new CSV files
events_df.to_csv('events_preprocessed.csv', index=False)
users_df.to_csv('users_preprocessed.csv', index=False)

#print("Data preprocessing complete. Preprocessed files saved as 'events_preprocessed.csv' and 'users_preprocessed.csv'.")