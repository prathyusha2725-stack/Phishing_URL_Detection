import pandas as pd

file_path = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())