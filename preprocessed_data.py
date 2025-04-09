# Preprocessing script (run once)
import pandas as pd

# Load and merge data
stations = pd.read_csv("gwl-stations.csv")[['Station_Code', 'Well_Name', 'LATITUDE', 'LONGITUDE', 'Elevation']]
daily = pd.read_csv("gwl-daily.csv", parse_dates=['MSMT_DATE']).rename(columns={'MSMT_DATE': 'Date'})

# Merge and save as Parquet
merged = daily.merge(stations, on='Station_Code')
merged.to_parquet("preprocessed_data.parquet")