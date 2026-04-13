import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_model_comparison(results):

    os.makedirs("outputs/plots", exist_ok=True)

    model_names = list(results.keys())

    r2_scores = [results[m]["R2"] for m in model_names]
    mae_scores = [results[m]["MAE"] for m in model_names]
    rmse_scores = [results[m]["RMSE"] for m in model_names]

    # R² plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=r2_scores)
    plt.title("Model Comparison - R² Score")
    plt.ylabel("R² Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/plots/model_r2_comparison.png")
    plt.close()

    # MAE plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=mae_scores)
    plt.title("Model Comparison - MAE")
    plt.ylabel("MAE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/plots/model_mae_comparison.png")
    plt.close()

    # RMSE plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=rmse_scores)
    plt.title("Model Comparison - RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/plots/model_rmse_comparison.png")
    plt.close()


def plot_actual_vs_predicted(model, X, y, targets):

    os.makedirs("outputs/plots", exist_ok=True)

    predictions = model.predict(X)

    for i, target in enumerate(targets):

        plt.figure(figsize=(6, 6))

        actual = y[target]

        if len(targets) == 1:
            predicted = predictions
        else:
            predicted = predictions[:, i]

        sns.scatterplot(x=actual, y=predicted)

        plt.plot(
            [actual.min(), actual.max()],
            [actual.min(), actual.max()],
            color="red",
            linestyle="--"
        )

        plt.title(f"Actual vs Predicted {target}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        plt.tight_layout()

        plt.savefig(
            f"outputs/plots/actual_vs_predicted_{target}.png"
        )

        plt.close()






def plot_speed_feed_heatmaps(results_df, targets):

    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("outputs/plots", exist_ok=True)

    for target in targets:

        pivot_table = results_df.pivot(
            index="Feed",
            columns="Speed",
            values=target
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="viridis",
            fmt=".3f"
        )

        plt.title(f"Speed vs Feed Heatmap for {target}")
        plt.xlabel("Speed")
        plt.ylabel("Feed")

        plt.tight_layout()

        plt.savefig(
            f"outputs/plots/speed_feed_heatmap_{target}.png"
        )

        plt.close()