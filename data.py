import pandas as pd

# Load the data file
data_file = "parkinsons.data"  # Replace with the actual file path

# Read the file into a pandas DataFrame
# Assuming the file uses a common delimiter (like a comma or tab)
try:
    df = pd.read_csv(data_file, sep=",")  # Change sep if needed, e.g., '\t' for tab-separated
except Exception as e:
    print(f"Error loading the file: {e}")
    exit()

# Save the DataFrame to a CSV file
csv_file = "parkinsons.csv"  # Desired output file name
df.to_csv(csv_file, index=False)

print(f"File converted successfully and saved as {csv_file}")
