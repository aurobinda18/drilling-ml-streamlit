import streamlit as st
import pandas as pd

from src.data_loader import load_data, load_material_properties
from src.preprocessing import split_features_targets
from src.train_models import train_models
from src.optimizer import (
    generate_parameter_grid,
    find_optimal_parameters,
    match_target_quality
)


st.set_page_config(
    page_title="Drilling Parameter Optimizer",
    layout="wide"
)

st.title("AI-Based Drilling Parameter Optimization Dashboard")

st.write(
    "Upload drilling dataset and material properties to begin analysis."
)


# Upload dataset section

drilling_file = st.file_uploader(
    "Upload Drilling Dataset (CSV)",
    type=["csv"]
)

material_file = st.file_uploader(
    "Upload Material Properties Dataset (CSV)",
    type=["csv"]
)


# Load datasets after upload

if drilling_file and material_file:

    drilling_df = pd.read_csv(drilling_file)
    material_df = pd.read_csv(material_file)

    df = drilling_df.merge(material_df, on="Material")

    st.success("Datasets loaded successfully")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    # Train models automatically

    X, y, targets = split_features_targets(df)

    models = train_models(X, y)

    st.success("Models trained successfully")

    st.write("Detected Targets:", targets)


    # Material selection dropdown (must stay INSIDE this block)

    available_materials = df["Material"].unique()

    selected_material = st.selectbox(
        "Select Material for Optimization",
        available_materials
    )

    st.write("Selected Material:", selected_material)

    st.subheader("Forward Optimization")

    if st.button("Find Optimal Parameters"):

        # Detect Speed and Feed ranges automatically
        speed_range = sorted(df["Speed"].unique())
        feed_range = sorted(df["Feed"].unique())

        # Extract selected material properties
        material_row = material_df[
            material_df["Material"] == selected_material
        ].iloc[0]

        diameter = df[
            df["Material"] == selected_material
        ]["Diameter"].iloc[0]

        youngs_modulus = material_row["Modulus (Automatic Young's) (MPa)"]
        max_stress = material_row["Maximum Stress (MPa)"]
        flex_modulus = material_row["Flex Modulus (MPa)"]

        # Generate parameter grid
        param_grid = generate_parameter_grid(
            speed_range,
            feed_range,
            diameter,
            youngs_modulus,
            max_stress,
            flex_modulus
        )

        # Select best model automatically
        best_model = list(models.values())[0]

        optimal_settings, full_results = find_optimal_parameters(
            best_model,
            param_grid,
            targets
        )

        st.success("Optimal Parameters Found")

        for target in targets:

            st.write(f"### Minimum {target}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Speed", optimal_settings[target]["Speed"])
            col2.metric("Feed", optimal_settings[target]["Feed"])
            col3.metric(
                f"Predicted {target}",
                round(optimal_settings[target][target], 4)
            )

    st.subheader("Reverse Optimization (Match Target Quality)")

    st.write("Enter desired output quality. Leave one blank if not needed.")

    col1, col2 = st.columns(2)

    target_ra = col1.number_input(
        "Target Surface Roughness (Ra)",
        min_value=0.0,
        step=0.01,
        format="%.3f"
    )

    target_fd = col2.number_input(
        "Target Delamination Factor (Fd)",
        min_value=0.0,
        step=0.01,
        format="%.3f"
    )

    if st.button("Recommend Machine Parameters"):

        # Convert empty inputs to None
        target_ra_value = target_ra if target_ra > 0 else None
        target_fd_value = target_fd if target_fd > 0 else None

        if target_ra_value is None and target_fd_value is None:

            st.warning("Please enter at least one target value.")

        else:

            # Detect Speed and Feed ranges
            speed_range = sorted(df["Speed"].unique())
            feed_range = sorted(df["Feed"].unique())

            # Extract selected material properties
            material_row = material_df[
                material_df["Material"] == selected_material
            ].iloc[0]

            diameter = df[
                df["Material"] == selected_material
            ]["Diameter"].iloc[0]

            youngs_modulus = material_row["Modulus (Automatic Young's) (MPa)"]
            max_stress = material_row["Maximum Stress (MPa)"]
            flex_modulus = material_row["Flex Modulus (MPa)"]

            param_grid = generate_parameter_grid(
                speed_range,
                feed_range,
                diameter,
                youngs_modulus,
                max_stress,
                flex_modulus
            )

            best_model = list(models.values())[0]

            best_match = match_target_quality(
                best_model,
                param_grid,
                targets,
                target_ra=target_ra_value,
                target_fd=target_fd_value
            )

            st.success("Recommended Parameters Found")

            col1, col2 = st.columns(2)

            col1.metric("Speed", best_match["Speed"])
            col2.metric("Feed", best_match["Feed"])

            if target_ra_value is not None and "Ra" in targets:
                st.metric("Predicted Ra", round(best_match["Ra"], 4))

            if target_fd_value is not None and "Fd" in targets:
                st.metric("Predicted Fd", round(best_match["Fd"], 4))