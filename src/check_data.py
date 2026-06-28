import pandas as pd

DATA_PATH = "data/forbes_global_2000_2026.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())