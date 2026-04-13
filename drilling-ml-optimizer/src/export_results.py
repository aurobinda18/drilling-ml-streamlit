

import pandas as pd
import os


def export_model_performance(results):

    os.makedirs("outputs/metrics", exist_ok=True)

    rows = []

    for model_name, metrics in results.items():

        rows.append({
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        "outputs/metrics/model_performance.csv",
        index=False
    )


def export_optimal_parameters(optimal_settings):

    os.makedirs("outputs/metrics", exist_ok=True)

    rows = []

    for target, values in optimal_settings.items():

        row = {
            "Target": target,
            "Speed": values["Speed"],
            "Feed": values["Feed"],
            f"Predicted_{target}": values[target]
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        "outputs/metrics/optimal_parameters.csv",
        index=False
    )


def export_reverse_optimization(best_match, targets):

    os.makedirs("outputs/metrics", exist_ok=True)

    row = {
        "Speed": best_match["Speed"],
        "Feed": best_match["Feed"]
    }

    if "Ra" in targets:
        row["Predicted_Ra"] = best_match["Ra"]

    if "Fd" in targets:
        row["Predicted_Fd"] = best_match["Fd"]

    df = pd.DataFrame([row])

    df.to_csv(
        "outputs/metrics/reverse_optimization_results.csv",
        index=False
    )