import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from skops.io import load, get_untrusted_types


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/model.skops"
DATA_PATH = "data/processed/eval.csv"
PLOT_DIR = "week9/plots"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

CLASS_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# Load model
# --------------------------------------------------

trusted = get_untrusted_types(file=MODEL_PATH)

model = load(
    MODEL_PATH,
    trusted=trusted
)


# --------------------------------------------------
# Load full evaluation dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]


# --------------------------------------------------
# SHAP full-dataset explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)


# --------------------------------------------------
# Display SHAP output information
# --------------------------------------------------

print("SHAP analysis completed.")
print(f"Dataset shape: {X.shape}")
print(f"SHAP values type: {type(shap_values)}")

if isinstance(shap_values, list):
    print(f"Number of class explanations: {len(shap_values)}")
    for i, values in enumerate(shap_values):
        print(
            f"Class {i} ({CLASS_NAMES[i]}): "
            f"shape={values.shape}"
        )
else:
    print(f"SHAP values shape: {shap_values.shape}")


# --------------------------------------------------
# Generate summary plots for all classes
# --------------------------------------------------

for class_index, class_name in enumerate(CLASS_NAMES):

    print(f"\nGenerating SHAP plot for {class_name}...")

    plt.figure()

    if isinstance(shap_values, list):

        shap.summary_plot(
            shap_values[class_index],
            X,
            show=False,
        )

    else:

        # Newer SHAP versions may return an Explanation
        # with an explicit output dimension.
        values = shap_values[:, :, class_index]

        shap.summary_plot(
            values,
            X,
            show=False,
        )

    plt.title(
        f"SHAP Summary Plot - {class_name.capitalize()}"
    )

    plt.tight_layout()

    output_path = os.path.join(
        PLOT_DIR,
        f"shap_{class_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


print("\nAll SHAP plots generated successfully.")
