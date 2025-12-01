import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helper: load pickle
# ------------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------------------------------------------------
# Helper: compute average confidence per question
# ------------------------------------------------------------
def avg_confidence(conf_list):
    if conf_list is None or len(conf_list) == 0:
        return 0.0
    return float(np.mean(conf_list))

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():

    # ----------------------------
    # File paths
    # ----------------------------
    som_pkl_path = "contexts/som_contexts_scores.pkl"
    cosine_pkl_path = "contexts/cosine_contexts_scores.pkl"
    eval_results_folder = "evaluation_results_log"

    # Automatically detect the CSV file
    csv_files = [f for f in os.listdir(eval_results_folder) if f.endswith(".csv")]
    
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV evaluation file found in evaluation_results_log/")
    if len(csv_files) > 1:
        print("Multiple CSVs found. Using:", csv_files[0])

    eval_csv_path = os.path.join(eval_results_folder, csv_files[0])

    # ----------------------------
    # Load data
    # ----------------------------
    print("Loading SOM and Cosine confidence files...")
    som_scores = load_pickle(som_pkl_path)
    cosine_scores = load_pickle(cosine_pkl_path)

    print("Loading evaluation results CSV...")
    df_eval = pd.read_csv(eval_csv_path)

    # Expecting the CSV to have:
    # som_contains_answer, cosine_contains_answer columns
    required_cols = ["som_contains_answer", "cosine_contains_answer"]
    for col in required_cols:
        if col not in df_eval.columns:
            raise ValueError(f"Missing required column '{col}' in evaluation CSV.")

    # ----------------------------
    # Convert lists of chunk confidences → average confidence per question
    # ----------------------------
    print("Computing average confidence per question...")

    df_eval["som_avg_conf"] = df_eval.index.map(
        lambda i: avg_confidence(som_scores.get(i, []))
    )

    df_eval["cosine_avg_conf"] = df_eval.index.map(
        lambda i: avg_confidence(cosine_scores.get(i, []))
    )

    # ----------------------------
    # Prepare labels and scores
    # ----------------------------
    y_som = df_eval["som_contains_answer"].astype(int).values
    y_cos = df_eval["cosine_contains_answer"].astype(int).values

    som_scores_arr = df_eval["som_avg_conf"].values
    cosine_scores_arr = df_eval["cosine_avg_conf"].values

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
