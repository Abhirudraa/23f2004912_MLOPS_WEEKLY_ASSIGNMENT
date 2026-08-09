import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/train.csv"
OUTPUT_DIR = "poisoned_data"

SEED = 42

POISON_LEVELS = [0, 5, 10, 50]

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "target"


# ---------------------------------------------------------
# Poisoning function
# ---------------------------------------------------------

def poison_dataset(df, poisoning_percentage, rng):
    """
    Replace a percentage of training samples with
    randomly generated features and random labels.
    """

    poisoned_df = df.copy()

    n_samples = len(poisoned_df)

    n_poison = int(round(
        n_samples * poisoning_percentage / 100
    ))

    # Randomly select samples to poison
    poison_indices = rng.choice(
        n_samples,
        size=n_poison,
        replace=False
    )

    # Generate random feature values
    #
    # We use ranges somewhat broader than the original
    # Iris feature ranges to simulate corrupted values.
    random_features = rng.uniform(
        low=0.0,
        high=8.0,
        size=(n_poison, len(FEATURE_COLUMNS))
    )

    # Random labels from the existing classes
    classes = df[TARGET_COLUMN].unique()

    random_labels = rng.choice(
        classes,
        size=n_poison
    )

    # Replace all four features
    poisoned_df.loc[
        poisoned_df.index[poison_indices],
        FEATURE_COLUMNS
    ] = random_features

    # Replace labels
    poisoned_df.loc[
        poisoned_df.index[poison_indices],
        TARGET_COLUMN
    ] = random_labels

    return poisoned_df, poison_indices


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading clean training dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Validate columns
    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level in POISON_LEVELS:

        rng = np.random.default_rng(SEED)

        poisoned_df, poison_indices = poison_dataset(
            df,
            level,
            rng
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            f"iris_poisoned_{level}.csv"
        )

        poisoned_df.to_csv(
            output_file,
            index=False
        )

        print("\n----------------------------------------")
        print(f"Poisoning level: {level}%")
        print(f"Poisoned samples: {len(poison_indices)}")
        print(f"Output: {output_file}")
        print("----------------------------------------")


if __name__ == "__main__":
    main()
