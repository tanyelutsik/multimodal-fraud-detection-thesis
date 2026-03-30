from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "paysim_clean.csv")

    with open(PROCESSED_DIR / "paysim_type_mapping.json", "r", encoding="utf-8") as f:
        type_mapping = json.load(f)

    reverse_type_mapping = {int(v): k for k, v in type_mapping.items()}
    return df, reverse_type_mapping


def plot_class_distribution(df: pd.DataFrame):
    class_counts = df["isFraud"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar")
    plt.xticks([0, 1], ["Non-Fraud", "Fraud"], rotation=0)
    plt.ylabel("Number of transactions")
    plt.title("Class distribution of the PaySim dataset")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "paysim_class_distribution.png", dpi=300)
    plt.show()


def plot_transaction_type_distribution(df: pd.DataFrame, reverse_type_mapping: dict):
    type_counts = df["type"].value_counts().sort_index()
    labels = [reverse_type_mapping[i] for i in type_counts.index]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, type_counts.values)
    plt.ylabel("Number of transactions")
    plt.title("Transaction type distribution in PaySim")
    plt.xticks(rotation=0)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "paysim_transaction_type_distribution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    df, reverse_type_mapping = load_data()
    print(df.shape)
    print(df.head())

    plot_class_distribution(df)
    plot_transaction_type_distribution(df, reverse_type_mapping)