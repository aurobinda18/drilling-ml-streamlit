from src.data_loader import load_data, load_material_properties
from src.preprocessing import split_features_targets
from src.train_models import train_models
from src.evaluate_models import evaluate_model
from src.reconstruction_check import reconstruction_test
from src.visualization import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_speed_feed_heatmaps
)

from src.optimizer import (
    generate_parameter_grid,
    find_optimal_parameters,
    match_target_quality
)

from src.export_results import (
    export_model_performance,
    export_optimal_parameters,
    export_reverse_optimization
)

def main():

    # Load dataset
    df = load_data(
        "data/drilling_data.csv",
        "data/material_properties.csv"
    )

    print("Dataset Loaded Successfully\n")

    # Split dataset
    X, y, targets = split_features_targets(df)

    print("Detected Targets:", targets)
    print("\nTraining Models...\n")

    models = train_models(X, y)

    print("Evaluating Models using LOOCV:\n")

    results = {}

    # Evaluate models
    for name, model in models.items():

        mae, rmse, r2 = evaluate_model(model, X, y)

        results[name] = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"{name}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print("-----------")

    # Select best model automatically
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_model = results[best_model_name]["model"]

    

    print(f"\nBest Model Selected: {best_model_name}")

# Export model performance table
    export_model_performance(results) 
    # Generate comparison plots
    plot_model_comparison(results)

    # Generate prediction accuracy plots
    plot_actual_vs_predicted(best_model, X, y, targets)

    print("\nVisualization plots saved in outputs/plots/")


    

    # Reconstruction accuracy check
    recon_error = reconstruction_test(best_model, X, y)

    print(f"Reconstruction RMSE: {recon_error:.6f}")

    print("\nRunning Optimization...\n")

    # Adjustable machine parameters
    # Automatically detect Speed and Feed ranges from dataset

    speed_range = sorted(df["Speed"].unique())
    feed_range = sorted(df["Feed"].unique())

    print("\nDetected Speed Range:", speed_range)
    print("Detected Feed Range:", feed_range)

    # Fixed material/tool properties (example: carbon)
   # Select material dynamically from dataset

    available_materials = df["Material"].unique()

    selected_material = available_materials[0]   # default selection (for script mode)

    print(f"\nMaterial Selected for Optimization: {selected_material}")

    material_df = load_material_properties("data/material_properties.csv")

    material_row = material_df[
        material_df["Material"] == selected_material
    ].iloc[0]


    # Automatically detect drill diameter from dataset
    diameter = df[df["Material"] == selected_material]["Diameter"].iloc[0]


    # Load mechanical properties dynamically
    youngs_modulus = material_row["Modulus (Automatic Young's) (MPa)"]
    max_stress = material_row["Maximum Stress (MPa)"]
    flex_modulus = material_row["Flex Modulus (MPa)"]

    # Generate parameter combinations
    param_grid = generate_parameter_grid(
        speed_range,
        feed_range,
        diameter,
        youngs_modulus,
        max_stress,
        flex_modulus
    )

    # Run optimization
    optimal_settings, full_results = find_optimal_parameters(
        best_model,
        param_grid,
        targets
    )


    # Generate optimization heatmaps
    plot_speed_feed_heatmaps(full_results, targets)

    print("Optimal Machine Settings:\n")
# Export optimal parameter recommendations
    export_optimal_parameters(optimal_settings)
                

    for target in targets:
        print(f"For Minimum {target}:")
        print(f"Speed: {optimal_settings[target]['Speed']}")
        print(f"Feed: {optimal_settings[target]['Feed']}")
        print(f"Predicted {target}: {optimal_settings[target][target]:.4f}")
        print("-----------")

    print("\nReverse Optimization (Match Target Quality)...\n")

    # Example desired targets (operator input simulation)
    target_ra = 1.5
    target_fd = None

    best_match = match_target_quality(
        best_model,
        param_grid,
        targets,
        target_ra=target_ra,
        target_fd=target_fd
    )

    print("Recommended Machine Settings for Desired Quality:\n")
# Export reverse optimization recommendations
    export_reverse_optimization(best_match, targets)
    if target_ra is not None:
        print(f"Target Ra: {target_ra}")

    if target_fd is not None:
        print(f"Target Fd: {target_fd}")

    print()

    print(f"Recommended Speed: {best_match['Speed']}")
    print(f"Recommended Feed: {best_match['Feed']}")

    if "Ra" in targets and target_ra is not None:
     print(f"Predicted Ra: {best_match['Ra']:.4f}")

    if "Fd" in targets and target_fd is not None:
     print(f"Predicted Fd: {best_match['Fd']:.4f}")

if __name__ == "__main__":
    main()