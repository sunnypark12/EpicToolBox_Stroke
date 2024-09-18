import pandas as pd

# Load the CSV file into a DataFrame
file_path = "/Users/sunho/Desktop/toolbox_final/SSST10_V1_TM_LG_FD_S2_t1.csv"  # Use your correct file path
data = pd.read_csv(file_path)

# Print the column names to ensure correct data loading
print(data.columns)
