# prepare_example_data.py
import pandas as pd

# Load the raw data files
df1 = pd.read_csv("temperature_data.csv")
df2 = pd.read_csv("weather_data.csv")

# Convert datetime columns
df1['datetime'] = pd.to_datetime(df1['datetime'])
df2['time'] = pd.to_datetime(df2['time'])

# Rename columns for consistency
df2 = df2.rename(columns={'time': 'datetime'})

# Merge the dataframes
combined_df = pd.merge(df1, df2, on='datetime', how='outer')

# Sort by datetime and set as index
combined_df = combined_df.sort_values('datetime')
combined_df = combined_df.set_index('datetime')

# Save the preprocessed data
combined_df.to_csv('preprocessed_building_data.csv')

print("Preprocessed data saved successfully.")