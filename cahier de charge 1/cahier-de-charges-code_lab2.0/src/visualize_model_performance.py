from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(workspace_root, "outputs", "all_company_metrics.csv")
    output_dir = os.path.join(workspace_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(metrics_path)

    # Top 15 best (lowest RMSE) per model
    top_xgb = df.nsmallest(15, "XGBOOST_RMSE")
    top_gb = df.nsmallest(15, "GBOOST_RMSE")

    plt.figure(figsize=(10, 6))
    plt.barh(top_xgb["COMPANY"], top_xgb["XGBOOST_RMSE"], color="#1f77b4")
    plt.title("Top 15 Companies by XGBoost RMSE (lower is better)")
    plt.xlabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top15_xgboost_rmse.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(top_gb["COMPANY"], top_gb["GBOOST_RMSE"], color="#ff7f0e")
    plt.title("Top 15 Companies by GradientBoosting RMSE (lower is better)")
    plt.xlabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top15_gboost_rmse.png"))
    plt.close()

    # Scatter comparison
    plt.figure(figsize=(7, 6))
    plt.scatter(df["XGBOOST_RMSE"], df["GBOOST_RMSE"], alpha=0.7)
    plt.plot([df["XGBOOST_RMSE"].min(), df["XGBOOST_RMSE"].max()],
             [df["XGBOOST_RMSE"].min(), df["XGBOOST_RMSE"].max()],
             linestyle="--", color="gray")
    plt.title("XGBoost vs GradientBoosting RMSE")
    plt.xlabel("XGBoost RMSE")
    plt.ylabel("GradientBoosting RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xgboost_vs_gboost_rmse.png"))
    plt.close()

    # Distributions
    plt.figure(figsize=(8, 5))
    plt.hist(df["XGBOOST_RMSE"], bins=30, alpha=0.7, label="XGBoost", color="#1f77b4")
    plt.hist(df["GBOOST_RMSE"], bins=30, alpha=0.7, label="GradientBoosting", color="#ff7f0e")
    plt.title("RMSE Distribution")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_distribution.png"))
    plt.close()

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
