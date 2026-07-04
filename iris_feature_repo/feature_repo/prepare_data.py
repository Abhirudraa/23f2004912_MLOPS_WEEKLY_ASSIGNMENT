import pandas as pd

def main():
    df = pd.read_csv("../../data/iris.csv")

    df["iris_id"] = range(len(df))
    df["event_timestamp"] = pd.Timestamp("2025-01-01")

    cols = [
        "iris_id",
        "event_timestamp",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    df = df[cols]

    df.to_parquet("data/iris_features.parquet", index=False)
    print("Saved iris_features.parquet")


if __name__ == "__main__":
    main()
