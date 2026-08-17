import os

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_PATH = "data/processed/train.csv"

PRODUCTION_PATH = "week9/production.csv"

PLOT_DIR = "week9/plots"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

DRIFT_FEATURE = "petal_length"
DRIFT_OFFSET = 1.0

ALPHA = 0.05


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)


# --------------------------------------------------
# Simulate production data
# --------------------------------------------------

production_df = train_df.copy()

production_df[DRIFT_FEATURE] = (
    production_df[DRIFT_FEATURE] + DRIFT_OFFSET
)

production_df.to_csv(
    PRODUCTION_PATH,
    index=False,
)

print(f"Saved simulated production data to: {PRODUCTION_PATH}")

print(
    f"\nApplied drift to {DRIFT_FEATURE}: "
    f"+{DRIFT_OFFSET}"
)


# --------------------------------------------------
# Statistical drift detection
# --------------------------------------------------

results = []

for feature in FEATURES:

    statistic, p_value = ks_2samp(
        train_df[feature],
        production_df[feature],
    )

    drift_detected = p_value < ALPHA

    results.append(
        {
            "feature": feature,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected,
        }
    )


results_df = pd.DataFrame(results)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nDrift Detection Results:")
print(results_df.to_string(index=False))


# --------------------------------------------------
# Distribution plots
# --------------------------------------------------

for feature in FEATURES:

    plt.figure(figsize=(8, 5))

    plt.hist(
        train_df[feature],
        bins=15,
        alpha=0.6,
        label="Training",
    )

    plt.hist(
        production_df[feature],
        bins=15,
        alpha=0.6,
        label="Production",
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(
        f"Training vs Production: {feature}"
    )

    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        PLOT_DIR,
        f"drift_{feature}.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


print("\nDrift analysis completed.")
