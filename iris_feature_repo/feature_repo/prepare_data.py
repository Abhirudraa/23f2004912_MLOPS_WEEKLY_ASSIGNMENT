import pandas as pd

# Read original IRIS dataset
df = pd.read_csv("../../data/iris.csv")

# Add unique entity id
df["iris_id"] = range(len(df))

# Feast requires an event timestamp
df["event_timestamp"] = pd.Timestamp("2025-01-01")

# Move columns to front
cols = [
    "iris_id",
    "event_timestamp",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

df = df[cols]

print(df.head())

# Save as Parquet
df.to_parquet(
    "data/iris_features.parquet",
    index=False
)

print("Saved iris_features.parquet")
