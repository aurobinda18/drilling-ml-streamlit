import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PLOTS_DIR = BASE_DIR / "outputs" / "plots"

def plot_model_comparison(results):

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    r2_scores = [results[m]["R2"] for m in models]
    mae_scores = [results[m]["MAE"] for m in models]
    rmse_scores = [results[m]["RMSE"] for m in models]

    sns.set_style("whitegrid")

    def save_plot(values, title, ylabel, filename, color):

        plt.figure(figsize=(7, 5))
        ax = sns.barplot(x=models, y=values, palette=color)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Models")

        for i, value in enumerate(values):
            ax.text(i, value, round(value, 4), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(str(OUTPUT_PLOTS_DIR / filename))
        plt.close()

    save_plot(
        r2_scores,
        "Model Comparison (R² Score)",
        "R² Score",
        "model_r2_comparison.png",
        "Blues"
    )

    save_plot(
        mae_scores,
        "Model Comparison (MAE)",
        "MAE",
        "model_mae_comparison.png",
        "Oranges"
    )

    save_plot(
        rmse_scores,
        "Model Comparison (RMSE)",
        "RMSE",
        "model_rmse_comparison.png",
        "Greens"
    )


def plot_actual_vs_predicted(model, X, y, targets):

    import numpy as np

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = np.asarray(model.predict(X))

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    for i, target in enumerate(targets):

        plt.figure(figsize=(6, 6))

        actual = np.asarray(y[target]).reshape(-1)

        # Models trained with a single target can still return (n, 1).
        target_idx = 0 if predictions.shape[1] == 1 else i
        predicted = predictions[:, target_idx].reshape(-1)

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

        plt.savefig(str(OUTPUT_PLOTS_DIR / f"actual_vs_predicted_{target}.png"))

        plt.close()






def plot_speed_feed_heatmaps(results_df, targets):

    import matplotlib.pyplot as plt
    import seaborn as sns

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")

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
            cmap="YlGnBu",
            linewidths=0.5,
            fmt=".3f"
        )

        optimal_row = results_df.loc[results_df[target].idxmin()]
        x_pos = list(pivot_table.columns).index(optimal_row["Speed"]) + 0.5
        y_pos = list(pivot_table.index).index(optimal_row["Feed"]) + 0.5

        plt.scatter(
            x_pos,
            y_pos,
            color="red",
            s=150,
            marker="X",
            label="Optimal Point"
        )

        plt.legend()

        plt.title(
            f"Effect of Speed and Feed on {target}",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Spindle Speed")
        plt.ylabel("Feed Rate")

        plt.tight_layout()

        plt.savefig(str(OUTPUT_PLOTS_DIR / f"speed_feed_heatmap_{target}.png"))

        plt.close()


def plot_3d_response_surface(results_df, targets):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for target in targets:

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        X = results_df["Speed"].values
        Y = results_df["Feed"].values
        Z = results_df[target].values

        ax.plot_trisurf(
            X,
            Y,
            Z,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9
        )

        ax.set_title(
            f"3D Response Surface: Speed vs Feed vs {target}",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel("Speed")
        ax.set_ylabel("Feed")
        ax.set_zlabel(target)

        plt.tight_layout()

        plt.savefig(str(OUTPUT_PLOTS_DIR / f"response_surface_{target}.png"))

        plt.close()


def plot_feature_importance(model, X):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract importance values
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_)
    else:
        return

    importance = np.asarray(importance)

    # Collapse multi-output importances/coefs into one score per feature.
    if importance.ndim > 1:
        importance = importance.mean(axis=0)

    importance = importance.reshape(-1)

    if len(importance) != len(X.columns):
        return

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    # Remove encoded material columns
    importance_df = importance_df[
        ~importance_df["Feature"].str.contains("Material")
    ]

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        palette="viridis"
    )

    plt.title(
        "Feature Importance Ranking (Machining Parameters Only)",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(str(OUTPUT_PLOTS_DIR / "feature_importance.png"))

    plt.close()


def generate_parameter_sensitivity_text(model, X):

    import numpy as np

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
        if importance.ndim > 1:
            importance = importance[0]
    else:
        return None

    feature_importance_pairs = list(zip(X.columns, importance))

    # Remove material-related encoded features
    feature_importance_pairs = [
        pair for pair in feature_importance_pairs
        if "Material" not in pair[0]
    ]

    feature_importance_pairs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    ranked_features = [
        feature for feature, _ in feature_importance_pairs
    ]

    return ranked_features