import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

# ------------------------------------------------------------
# Helper: load pickle
# ------------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    # ----------------------------
    # Parse command-line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(
        description="Generate ROC curves comparing SOM vs Cosine RAG methods"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="scaled_evaluation_results_5000_questions.csv",
        help="Evaluation results CSV file name or full path (default: scaled_evaluation_results_5000_questions.csv)"
    )
    args = parser.parse_args()

    # ----------------------------
    # File paths
    # ----------------------------
    eval_csv_directory = "evaluation_results_logs"
    # If the argument contains a path separator, use it as-is; otherwise prepend the directory
    if os.path.sep in args.csv or "/" in args.csv:
        eval_csv_path = args.csv
    else:
        eval_csv_path = os.path.join(eval_csv_directory, args.csv)

    # Verify the file exists
    if not os.path.exists(eval_csv_path):
        raise FileNotFoundError(f"CSV file not found at: {eval_csv_path}")

    print("Loading evaluation results CSV...")
    df_eval = pd.read_csv(eval_csv_path)

    # Expecting the CSV to have:
    # som_contains_answer, cosine_contains_answer columns
    required_cols = ["som_contains_answer", "cosine_contains_answer", "som_avg_confidence", "cosine_avg_confidence"]
    for col in required_cols:
        if col not in df_eval.columns:
            raise ValueError(f"Missing required column '{col}' in evaluation CSV.")

    # ----------------------------
    # Prepare labels and scores
    # ----------------------------
    y_som = df_eval["som_contains_answer"].astype(int).values
    y_cos = df_eval["cosine_contains_answer"].astype(int).values

    som_scores_arr = df_eval["som_avg_confidence"].values
    cosine_scores_arr = df_eval["cosine_avg_confidence"].values

    # ----------------------------
    # Compute ROC curves
    # ----------------------------
    print("Computing ROC curves...")

    fpr_som, tpr_som, _ = roc_curve(y_som, som_scores_arr)
    auc_som = auc(fpr_som, tpr_som)

    fpr_cos, tpr_cos, _ = roc_curve(y_cos, cosine_scores_arr)
    auc_cos = auc(fpr_cos, tpr_cos)

    # ----------------------------
    # Plot ROC curves
    # ----------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_som, tpr_som, label=f"SOM RAG (AUC = {auc_som:.3f})")
    plt.plot(fpr_cos, tpr_cos, label=f"Non-SOM (Cosine) RAG (AUC = {auc_cos:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: SOM RAG vs Non-SOM (Cosine) RAG")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot to evaluation_results_logs folder
    plot_path = os.path.join(eval_csv_directory, "roc_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"ROC curve saved to: {plot_path}")

    plt.show()

    print("\n===== ROC AUC RESULTS =====")
    print(f"SOM RAG AUC       : {auc_som:.4f}")
    print(f"Cosine RAG AUC    : {auc_cos:.4f}")
    print("============================\n")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
