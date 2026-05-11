import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_data, load_material_properties
from src.preprocessing import split_features_targets
from src.train_models import train_models
from src.evaluate_models import evaluate_model
from src.visualization import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_speed_feed_heatmaps,
    plot_3d_response_surface,
    plot_feature_importance,
    generate_parameter_sensitivity_text
)
from src.optimizer import (
    generate_parameter_grid,
    find_optimal_parameters,
    match_target_quality
)
from src.project_insights import (
    build_project_summary,
    summary_to_text,
    summary_to_flat_csv_df,
    summary_to_json_str,
    summary_to_docx_bytes
)


st.set_page_config(
    page_title="Drilling Parameter Optimizer",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"

# Theme selector
theme_mode = st.sidebar.selectbox(
    "Select UI Theme",
    ["Academic", "Jarvis", "Industrial", "Minimal"]
)

accent_color = st.sidebar.color_picker(
    "Accent Color",
    "#ff4b4b"
)

workspace_bg = st.sidebar.selectbox(
    "Workspace Background",
    [
        "None",
        "Cross Grid",
        "Blueprint Grid",
        "Engineering Dots",
        "Neon HUD Grid",
        "Diagonal Mesh",
        "Soft Graph Paper"
    ]
)


def apply_advanced_theme(theme, accent):

    if theme == "Jarvis":

        st.markdown(f"""
        <style>

        html, body, .stApp {{
            background: radial-gradient(circle at center,
            #0b0f19, #02040a);
            color: white;
        }}

        section.main > div {{
            backdrop-filter: blur(14px);
        }}

        .stButton>button {{
            background: {accent};
            color: black;
            border-radius: 14px;
            box-shadow: 0px 0px 12px {accent};
        }}

        .stMetric {{
            background: rgba(255,255,255,0.05);
            border: 1px solid {accent};
            border-radius: 16px;
            padding: 14px;
            backdrop-filter: blur(12px);
        }}

        </style>
        """, unsafe_allow_html=True)


    elif theme == "Industrial":

        st.markdown(f"""
        <style>

        html, body, .stApp {{
            background-color: #111111;
            color: #e0e0e0;
        }}

        .stButton>button {{
            background: {accent};
            border-radius: 4px;
            font-weight: bold;
        }}

        .stMetric {{
            background: #1e1e1e;
            border-left: 5px solid {accent};
            padding: 12px;
        }}

        </style>
        """, unsafe_allow_html=True)


    elif theme == "Minimal":

        st.markdown(f"""
        <style>

        html, body, .stApp {{
            background: white;
            color: black;
        }}

        .stButton>button {{
            background: {accent};
            border-radius: 8px;
        }}

        </style>
        """, unsafe_allow_html=True)


    else:  # Academic default

        st.markdown(f"""
        <style>

        html, body, .stApp {{
            background: #f4f7fb;
        }}

        h1, h2, h3 {{
            color: {accent};
        }}

        .stButton>button {{
            background: {accent};
            color: white;
        }}

        </style>
        """, unsafe_allow_html=True)


apply_advanced_theme(theme_mode, accent_color)


def apply_workspace_background(style):

    if style == "None":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: none !important;
            background-size: auto !important;
            background-color: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)

    elif style == "Cross Grid":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-image:
                linear-gradient(#d0d7e2 1px, transparent 1px),
                linear-gradient(90deg, #d0d7e2 1px, transparent 1px) !important;
            background-size: 35px 35px !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)


    elif style == "Blueprint Grid":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0a1f44 !important;
            background-image:
                linear-gradient(#1e4fa3 1px, transparent 1px),
                linear-gradient(90deg, #1e4fa3 1px, transparent 1px) !important;
            background-size: 40px 40px !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)


    elif style == "Engineering Dots":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-image:
                radial-gradient(#c7cedb 1px, transparent 1px) !important;
            background-size: 24px 24px !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)


    elif style == "Neon HUD Grid":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #020617 !important;
            background-image:
                linear-gradient(#00eaff33 1px, transparent 1px),
                linear-gradient(90deg, #00eaff33 1px, transparent 1px) !important;
            background-size: 38px 38px !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)


    elif style == "Diagonal Mesh":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-image:
                repeating-linear-gradient(
                    45deg,
                    #e6ecf5,
                    #e6ecf5 1px,
                    transparent 1px,
                    transparent 18px
                ) !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)


    elif style == "Soft Graph Paper":

        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-image:
                linear-gradient(#eef2f7 1px, transparent 1px),
                linear-gradient(90deg, #eef2f7 1px, transparent 1px) !important;
            background-size: 22px 22px !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)


apply_workspace_background(workspace_bg)

st.markdown("""
<style>

.stButton>button:hover {
transform: scale(1.06);
transition: 0.2s ease-in-out;
}

[data-testid="stFileUploader"] {
    border-radius: 12px;
    border: 2px dashed #1f4e79;
    padding: 8px;
}

</style>
""", unsafe_allow_html=True)

st.title("AI-Based Drilling Parameter Optimization Dashboard")

st.markdown(f"""
<div style="
background: linear-gradient(90deg,{accent_color},transparent);
padding:14px;
border-radius:14px;
font-size:20px;
font-weight:600;
letter-spacing:1px;
">
AI DRILLING PARAMETER OPTIMIZATION CONSOLE
</div>
""", unsafe_allow_html=True)

# Sidebar control panel
st.sidebar.markdown(f"""
<div style="
background:{accent_color};
padding:10px;
border-radius:10px;
font-weight:bold;
color:white;
">
CONTROL PANEL
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Workflow")

st.sidebar.markdown("""
1. Upload dataset  
2. Select targets  
3. Choose prediction / optimization mode  
4. View results  
""")

st.write(
    "Upload drilling dataset and material properties to begin analysis."
)


def align_param_grid_with_training_features(param_grid, feature_columns):
    """Match prediction input columns to the exact feature set used in training."""

    aligned_grid = param_grid.copy()

    for col in feature_columns:
        if col not in aligned_grid.columns:
            aligned_grid[col] = 0

    aligned_grid = aligned_grid[feature_columns]

    return aligned_grid


def get_column_unit(column_name):
    """Return a display unit for known machining columns."""

    unit_map = {
        "Speed": "rpm",
        "Feed": "mm/min",
        "Diameter": "mm",
        "Ra": "um",
        "Fd": "ratio",
        "Surface_Roughness": "um",
        "Delamination": "ratio",
        "Force": "N",
        "Temperature": "degC",
        "Wear": "mm",
        "Modulus (Automatic Young's) (MPa)": "MPa",
        "Maximum Stress (MPa)": "MPa",
        "Flex Modulus (MPa)": "MPa"
    }

    if column_name in unit_map:
        return unit_map[column_name]

    if "_" in column_name:
        base_name = column_name.split("_", 1)[0]
        return unit_map.get(base_name, None)

    return None


def format_label_with_unit(column_name):
    """Format a UI label with unit if available."""

    unit = get_column_unit(column_name)

    if unit:
        return f"{column_name} ({unit})"

    return column_name


def to_float_or_none(value):
    """Convert value to float when possible, otherwise return None."""

    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(parsed):
        return None

    return float(parsed)


def extract_material_properties(selected_material, material_df, df):
    """Collect all numeric material properties for the selected material."""

    if selected_material is None:
        return {}

    property_values = {}
    material_sources = []

    if material_df is not None:
        material_sources.append(material_df)

    material_sources.append(df)

    for source_df in material_sources:

        if source_df is None or "Material" not in source_df.columns:
            continue

        material_rows = source_df[source_df["Material"] == selected_material]

        if material_rows.empty:
            continue

        material_row = material_rows.iloc[0]

        for col, value in material_row.items():
            if col == "Material":
                continue

            numeric_value = to_float_or_none(value)

            if numeric_value is None:
                continue

            if col not in property_values:
                property_values[col] = numeric_value

    return property_values


def normalize_uploaded_dataframe(df):
    """Normalize uploaded CSV data for reliable numeric handling in live runs."""

    normalized = df.copy()

    # Remove accidental spaces from column names (e.g., " Ra ").
    normalized.columns = [str(col).strip() for col in normalized.columns]

    for col in normalized.columns:
        series = normalized[col]

        # Trim string cells and try locale-tolerant numeric parsing.
        if series.dtype == "object":
            stripped = series.astype(str).str.strip()
            parsed = pd.to_numeric(
                stripped.str.replace(",", ".", regex=False),
                errors="coerce"
            )

            # Convert when almost all non-empty values are numeric.
            non_empty = stripped.ne("").sum()
            numeric = parsed.notna().sum()
            if non_empty > 0 and (numeric / non_empty) >= 0.95:
                normalized[col] = parsed
            else:
                normalized[col] = stripped

    return normalized


def load_optional_csv(csv_path):
    """Load a CSV file if it exists, otherwise return None."""

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return None


# Upload dataset section (in sidebar)

st.sidebar.markdown("### Upload Data")

drilling_file = st.sidebar.file_uploader(
    "Drilling Dataset",
    type=["csv"]
)

material_file = st.sidebar.file_uploader(
    "Material Dataset (Optional)",
    type=["csv"]
)


# Load datasets after upload

if drilling_file:

    drilling_df = normalize_uploaded_dataframe(pd.read_csv(drilling_file))

    if material_file:

        material_df = normalize_uploaded_dataframe(pd.read_csv(material_file))

        df = drilling_df.merge(material_df, on="Material")

        st.success("Datasets merged successfully")

    else:

        df = drilling_df.copy()

        material_df = None

        st.warning("Material properties file not provided. Running without material features.")

    # Train models automatically

    X, y, targets = split_features_targets(df)
    feature_columns = list(X.columns)

    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
<div style="padding:15px;background:#e3f2fd;border-radius:10px;">
<b>Rows</b><br>{df.shape[0]}
</div>
""", unsafe_allow_html=True)

    col2.markdown(f"""
<div style="padding:15px;background:#e8f5e9;border-radius:10px;">
<b>Columns</b><br>{df.shape[1]}
</div>
""", unsafe_allow_html=True)

    col3.markdown(f"""
<div style="padding:15px;background:#fff3e0;border-radius:10px;">
<b>Targets</b><br>{len(targets)}
</div>
""", unsafe_allow_html=True)

    present_unit_columns = [
        col for col in df.columns
        if get_column_unit(col) is not None
    ]

    if present_unit_columns:
        unit_lines = [
            f"- {col}: {get_column_unit(col)}"
            for col in present_unit_columns
        ]
        st.info("Detected Units\n" + "\n".join(unit_lines))

    st.markdown("""
<hr style="border:1px solid #d0d7de;">
""", unsafe_allow_html=True)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    # Update dataset summary with detected targets
    col_targets = st.columns(1)
    col_targets[0].metric("Targets Detected", len(targets))

    models = train_models(X, y)

    # Evaluate models and generate comparison plots
    results = {}

    for name, model in models.items():

        mae, rmse, r2 = evaluate_model(model, X, y)

        results[name] = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    performance_rows = []
    for name, payload in results.items():
        performance_rows.append(
            {
                "Model": name,
                "MAE": payload["MAE"],
                "RMSE": payload["RMSE"],
                "R2": payload["R2"]
            }
        )

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(performance_rows).to_csv(
        str(OUTPUT_METRICS_DIR / "model_performance.csv"),
        index=False
    )

    # Select best model automatically
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_r2 = results[best_model_name]["R2"]

    if best_r2 > 0.95:
        confidence_level = "Excellent"
    elif best_r2 > 0.90:
        confidence_level = "High"
    elif best_r2 > 0.80:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    st.sidebar.success(f"Prediction Confidence: {confidence_level}")

    best_model = results[best_model_name]["model"]
    best_model.fit(X, y)

    if "project_summary" not in st.session_state:
        st.session_state.project_summary = None

    if "forward_optimization_full_results" not in st.session_state:
        st.session_state.forward_optimization_full_results = None

    if "forward_optimization_optimal_settings" not in st.session_state:
        st.session_state.forward_optimization_optimal_settings = None

    if "reverse_optimization_results" not in st.session_state:
        st.session_state.reverse_optimization_results = None

    # Generate plots
    plot_model_comparison(results)
    plot_actual_vs_predicted(best_model, X, y, targets)

    plot_feature_importance(best_model, X)

    st.sidebar.markdown("### Dataset Health")
    st.sidebar.success(f"Rows: {df.shape[0]}")
    st.sidebar.info(f"Features: {X.shape[1]}")
    st.sidebar.warning(f"Targets: {len(targets)}")

    st.success("Models trained successfully")

    st.write("Detected Targets:", targets)

    st.info(
        f"Best model automatically selected based on highest R² score for targets: {targets}"
    )

    # Let user choose which output variable(s) to optimize
    selected_targets = st.multiselect(
        "Select Output Variable(s) to Optimize",
        targets,
        default=targets
    )

    if len(selected_targets) == 0:
        st.warning("Please select at least one target variable.")

    # Material selection dropdown (must stay INSIDE this block)

    if "Material" in df.columns:

        available_materials = df["Material"].unique()

        selected_material = st.selectbox(
            "Select Material for Optimization",
            available_materials
        )

    else:

        selected_material = None

    material_properties = extract_material_properties(
        selected_material,
        material_df,
        df
    )

    st.write("Selected Material:", selected_material)

    st.markdown("### Selected Prediction Model")

    st.success("Best performing model selected automatically.")

    # Create dashboard tabs
    tab_prediction, tab_forward_opt, tab_reverse_opt, tab_analysis, tab_project_insights = st.tabs([
        "Prediction",
        "Forward Optimization",
        "Reverse Optimization",
        "Model Analysis",
        "Project Insights and Results Export"
    ])

    # ====== PREDICTION TAB ======
    with tab_prediction:

        st.markdown("""
<hr style="border:1px solid #d0d7de;">
""", unsafe_allow_html=True)
        st.subheader("Prediction Mode")

        st.write("Enter machining parameters to predict output values.")

        diameter_candidates = []

        if "Diameter" in df.columns:
            diameter_candidates = pd.to_numeric(
                df["Diameter"],
                errors="coerce"
            ).dropna().tolist()

        diameter = float(diameter_candidates[0]) if diameter_candidates else 0.0
        youngs_modulus = material_properties.get("Modulus (Automatic Young's) (MPa)", 0.0)
        max_stress = material_properties.get("Maximum Stress (MPa)", 0.0)
        flex_modulus = material_properties.get("Flex Modulus (MPa)", 0.0)

        st.markdown("### Adjustable Machining Parameters")

        speed_value = st.slider(
            "Speed",
            min_value=float(df["Speed"].min()),
            max_value=float(df["Speed"].max()),
            value=float(df["Speed"].mean())
        )

        feed_value = st.slider(
            "Feed",
            min_value=float(df["Feed"].min()),
            max_value=float(df["Feed"].max()),
            value=float(df["Feed"].mean())
        )

        st.markdown("### Material Properties")

        if material_properties:
            property_items = list(material_properties.items())
            display_columns = st.columns(min(3, len(property_items)))

            for idx, (prop_name, prop_value) in enumerate(property_items):
                unit = get_column_unit(prop_name)
                value_text = f"{prop_value:.2f} {unit}" if unit else f"{prop_value:.2f}"
                display_columns[idx % len(display_columns)].info(
                    f"{prop_name}\n\n{value_text}"
                )
        else:
            st.info("No numeric material properties found for selected material.")

        input_row = {
            "Speed": speed_value,
            "Feed": feed_value,
            "Diameter": diameter,
            "Material": selected_material,
        }
        input_row.update(material_properties)

        input_df = pd.DataFrame([input_row])

        encoded_input_df = pd.get_dummies(input_df, drop_first=True)

        for column in X.columns:
            if column not in encoded_input_df.columns:
                encoded_input_df[column] = 0

        encoded_input_df = encoded_input_df[X.columns]

        prediction = best_model.predict(encoded_input_df)
        prediction_array = np.asarray(prediction)

        st.markdown("### Live Prediction Output")

        if len(selected_targets) == 1:

            value = round(float(prediction_array.reshape(-1)[0]), 4)
            target = format_label_with_unit(selected_targets[0])
            st.markdown(f"""
<div style="
background: rgba(255,255,255,0.05);
border: 1px solid {accent_color};
padding:20px;
border-radius:18px;
text-align:center;
font-size:22px;
">
<b>{target}</b><br>{value}
</div>
""", unsafe_allow_html=True)

        else:

            metric_cols = st.columns(len(selected_targets))

            if prediction_array.ndim == 1:
                prediction_array = prediction_array.reshape(1, -1)

            for i, target in enumerate(selected_targets):
                value = round(float(prediction_array[0][i]), 4)
                metric_cols[i].markdown(f"""
<div style="
background: rgba(255,255,255,0.05);
border: 1px solid {accent_color};
padding:20px;
border-radius:18px;
text-align:center;
font-size:22px;
">
<b>{target}</b><br>{value}
</div>
""", unsafe_allow_html=True)

        st.download_button(
            "Download Prediction Result",
            input_df.to_csv(index=False),
            file_name="prediction_input.csv"
        )

    # ====== FORWARD OPTIMIZATION TAB ======
    with tab_forward_opt:

        st.markdown("""
<hr style="border:1px solid #d0d7de;">
""", unsafe_allow_html=True)
        st.subheader("Forward Optimization")

        if st.button("Find Optimal Parameters"):

            # Detect Speed and Feed ranges automatically
            speed_range = sorted(df["Speed"].unique())
            feed_range = sorted(df["Feed"].unique())

            # Detect diameter automatically if present
            if "Diameter" in df.columns:

                if selected_material is not None and "Material" in df.columns:

                    diameter = df[
                        df["Material"] == selected_material
                    ]["Diameter"].iloc[0]

                else:

                    diameter = df["Diameter"].iloc[0]

            else:

                diameter = None

            # Load material properties dynamically for selected material.
            youngs_modulus = material_properties.get(
                "Modulus (Automatic Young's) (MPa)",
                0.0
            )

            max_stress = material_properties.get(
                "Maximum Stress (MPa)",
                0.0
            )

            flex_modulus = material_properties.get(
                "Flex Modulus (MPa)",
                0.0
            )

            # Generate parameter grid
            param_grid = generate_parameter_grid(
                speed_range,
                feed_range,
                diameter if diameter is not None else 0,
                youngs_modulus,
                max_stress,
                flex_modulus
            )

            for prop_name, prop_value in material_properties.items():
                if prop_name not in param_grid.columns:
                    param_grid[prop_name] = prop_value

            param_grid = align_param_grid_with_training_features(
                param_grid,
                feature_columns
            )

            # Select best model automatically
            optimal_settings, full_results = find_optimal_parameters(
                best_model,
                param_grid,
                selected_targets
            )

            plot_speed_feed_heatmaps(full_results, selected_targets)

            # Generate 3D response surface plots
            plot_3d_response_surface(full_results, selected_targets)

            OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
            full_results.to_csv(
                str(OUTPUT_METRICS_DIR / "full_prediction_surface.csv"),
                index=False
            )

            st.success("Optimal Parameters Found")

            st.success(
                "Highlighted region in heatmap corresponds to minimum predicted output zone."
            )

            for target in selected_targets:

                st.write(f"### Minimum {target}")

                col1, col2, col3 = st.columns(3)

                col1.metric(format_label_with_unit("Speed"), optimal_settings[target]["Speed"])
                col2.metric(format_label_with_unit("Feed"), optimal_settings[target]["Feed"])
                col3.metric(
                    f"Predicted {format_label_with_unit(target)}",
                    round(optimal_settings[target][target], 4)
                )

            st.session_state.forward_optimization_full_results = full_results
            st.session_state.forward_optimization_optimal_settings = optimal_settings

            optimal_rows = []
            for target, values in optimal_settings.items():
                row = {
                    "Target": target,
                    "Speed": values["Speed"],
                    "Feed": values["Feed"],
                    f"Predicted_{target}": values[target]
                }
                optimal_rows.append(row)

            pd.DataFrame(optimal_rows).to_csv(
                str(OUTPUT_METRICS_DIR / "optimal_parameters.csv"),
                index=False
            )

            st.download_button(
                "Download Optimization Results",
                full_results.to_csv(index=False),
                file_name="optimization_results.csv"
            )

    # ====== REVERSE OPTIMIZATION TAB ======
    with tab_reverse_opt:

        st.markdown("""
<hr style="border:1px solid #d0d7de;">
""", unsafe_allow_html=True)
        st.subheader("Reverse Optimization (Match Target Quality)")

        st.write("Enter desired output quality. Leave one blank if not needed.")

        col1, col2 = st.columns(2)

        target_ra = col1.number_input(
            f"Target {format_label_with_unit('Ra')}",
            min_value=0.0,
            step=0.01,
            format="%.3f"
        )

        target_fd = col2.number_input(
            f"Target {format_label_with_unit('Fd')}",
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

                # Detect diameter automatically if present
                if "Diameter" in df.columns:

                    if selected_material is not None and "Material" in df.columns:

                        diameter = df[
                            df["Material"] == selected_material
                        ]["Diameter"].iloc[0]

                    else:

                        diameter = df["Diameter"].iloc[0]

                else:

                    diameter = None

                # Load material properties dynamically for selected material.
                youngs_modulus = material_properties.get(
                    "Modulus (Automatic Young's) (MPa)",
                    0.0
                )

                max_stress = material_properties.get(
                    "Maximum Stress (MPa)",
                    0.0
                )

                flex_modulus = material_properties.get(
                    "Flex Modulus (MPa)",
                    0.0
                )

                param_grid = generate_parameter_grid(
                    speed_range,
                    feed_range,
                    diameter if diameter is not None else 0,
                    youngs_modulus,
                    max_stress,
                    flex_modulus
                )

                for prop_name, prop_value in material_properties.items():
                    if prop_name not in param_grid.columns:
                        param_grid[prop_name] = prop_value

                param_grid = align_param_grid_with_training_features(
                    param_grid,
                    feature_columns
                )

                best_match = match_target_quality(
                    best_model,
                    param_grid,
                    selected_targets,
                    target_ra=target_ra_value,
                    target_fd=target_fd_value
                )

                st.success("Recommended Parameters Found")

                col1, col2 = st.columns(2)

                col1.metric(format_label_with_unit("Speed"), best_match["Speed"])
                col2.metric(format_label_with_unit("Feed"), best_match["Feed"])

                if target_ra_value is not None and "Ra" in selected_targets:
                    st.metric(f"Predicted {format_label_with_unit('Ra')}", round(best_match["Ra"], 4))

                if target_fd_value is not None and "Fd" in selected_targets:
                    st.metric(f"Predicted {format_label_with_unit('Fd')}", round(best_match["Fd"], 4))

                reverse_result_df = pd.DataFrame([best_match.to_dict()])
                st.session_state.reverse_optimization_results = reverse_result_df
                reverse_result_df.to_csv(
                    str(OUTPUT_METRICS_DIR / "reverse_optimization_results.csv"),
                    index=False
                )

    # ====== MODEL ANALYSIS TAB ======
    with tab_analysis:

        st.markdown("""
<hr style="border:1px solid #d0d7de;">
""", unsafe_allow_html=True)
        st.subheader("Model Performance Analysis")

        plot_dir = OUTPUT_PLOTS_DIR

        st.subheader("Model Performance Comparison")

        col1, col2, col3 = st.columns(3)

        if (plot_dir / "model_r2_comparison.png").exists():
            col1.image(str(plot_dir / "model_r2_comparison.png"), use_container_width=True)
        else:
            col1.info("R² comparison plot not available yet.")

        if (plot_dir / "model_mae_comparison.png").exists():
            col2.image(str(plot_dir / "model_mae_comparison.png"), use_container_width=True)
        else:
            col2.info("MAE comparison plot not available yet.")

        if (plot_dir / "model_rmse_comparison.png").exists():
            col3.image(str(plot_dir / "model_rmse_comparison.png"), use_container_width=True)
        else:
            col3.info("RMSE comparison plot not available yet.")

        st.subheader("Prediction Accuracy")

        st.subheader("Feature Influence Analysis")

        if (plot_dir / "feature_importance.png").exists():
            st.image(
                str(plot_dir / "feature_importance.png"),
                caption="Relative Importance of Input Parameters",
                use_container_width=True
            )
        else:
            st.info("Feature importance chart not available yet.")

        ranked_features = generate_parameter_sensitivity_text(
            best_model,
            X
        )

        if ranked_features:

            st.subheader("Parameter Sensitivity Interpretation")

            st.success(
                f"Most influential parameter: {ranked_features[0]}"
            )

            if len(ranked_features) > 1:

                st.info(
                    f"Second most influential parameter: {ranked_features[1]}"
                )

            if len(ranked_features) > 2:

                st.info(
                    f"Third most influential parameter: {ranked_features[2]}"
                )

            st.markdown("""
These rankings indicate which machining parameters most strongly affect the predicted output.
Higher-ranked parameters contribute more significantly to drilling performance variation.
""")

        st.info(
            """
This chart ranks how strongly each machining parameter affects predicted output quality.
Higher importance means greater influence on drilling performance.
"""
        )

        if selected_targets:
            for target in selected_targets:
                plot_path = plot_dir / f"actual_vs_predicted_{target}.png"

                if plot_path.exists():
                    st.image(str(plot_path), caption=f"Actual vs Predicted {target}", use_container_width=True)
                else:
                    st.info(f"Actual vs Predicted plot not available for {target}.")

        st.subheader("Optimization Heatmaps")

        if selected_targets:
            for target in selected_targets:
                plot_path = plot_dir / f"speed_feed_heatmap_{target}.png"

                if plot_path.exists():
                    st.image(str(plot_path), caption=f"Speed vs Feed Heatmap for {target}", use_container_width=True)
                else:
                    st.info(f"Heatmap not available for {target}.")

        st.subheader("3D Response Surface Visualization")

        for target in selected_targets:

            plot_path = plot_dir / f"response_surface_{target}.png"

            if plot_path.exists():
                st.image(
                    str(plot_path),
                    caption=f"3D Response Surface for {target}",
                    use_container_width=True
                )
            else:
                st.info(f"3D response surface not available for {target}.")

        st.markdown("### How to Interpret Heatmaps")

        st.info(
            """
Heatmaps show how machining parameters affect output quality:

• Each cell represents predicted output for a Speed-Feed combination  
• Darker regions = higher output values  
• Lighter regions = lower output values  
• The optimal machining region appears in the lightest zone  

Example:
If minimizing surface roughness (Ra), select Speed-Feed values from the lightest region.
"""
        )

    # ====== PROJECT INSIGHTS AND RESULTS EXPORT TAB ======
    with tab_project_insights:

        st.markdown("""
<hr style="border:1px solid #d0d7de;">
""", unsafe_allow_html=True)
        st.subheader("Project Insights and Results Export")

        st.write(
            "Generate a concise, evidence-based project summary from the actual trained models, optimization outputs, and saved visualizations."
        )

        forward_results_df = st.session_state.get("forward_optimization_full_results")
        if forward_results_df is None:
            forward_results_df = load_optional_csv(OUTPUT_METRICS_DIR / "full_prediction_surface.csv")

        reverse_results_df = st.session_state.get("reverse_optimization_results")
        if reverse_results_df is None:
            reverse_results_df = load_optional_csv(OUTPUT_METRICS_DIR / "reverse_optimization_results.csv")

        selected_targets_for_summary = selected_targets if len(selected_targets) > 0 else targets

        if st.button("Generate Project Summary"):
            project_summary = build_project_summary(
                df=df,
                X=X,
                y=y,
                targets=targets,
                results=results,
                best_model=best_model,
                selected_targets=selected_targets_for_summary,
                full_results=forward_results_df,
                optimal_settings=st.session_state.get("forward_optimization_optimal_settings"),
                reverse_results=reverse_results_df,
            )

            st.session_state.project_summary = project_summary

            flat_csv_df = summary_to_flat_csv_df(project_summary)
            flat_csv_df.to_csv(
                str(OUTPUT_METRICS_DIR / "project_summary.csv"),
                index=False
            )

            with open(OUTPUT_METRICS_DIR / "project_summary.txt", "w", encoding="utf-8") as file_handle:
                file_handle.write(summary_to_text(project_summary))

            with open(OUTPUT_METRICS_DIR / "project_summary.json", "w", encoding="utf-8") as file_handle:
                file_handle.write(summary_to_json_str(project_summary))

            docx_bytes = summary_to_docx_bytes(project_summary)
            if docx_bytes is not None:
                with open(OUTPUT_METRICS_DIR / "project_summary.docx", "wb") as file_handle:
                    file_handle.write(docx_bytes)

            st.success("Project summary generated from the current project outputs.")

        if st.session_state.project_summary is not None:
            project_summary = st.session_state.project_summary

            st.subheader("Dataset Summary")
            dataset_summary_df = pd.DataFrame(
                [
                    ["Total samples", project_summary["dataset_summary"]["total_samples"]],
                    ["Input features", project_summary["dataset_summary"]["input_features"]],
                    ["Output features", project_summary["dataset_summary"]["output_features"]],
                    ["Feature names", ", ".join(project_summary["dataset_summary"]["feature_names"])],
                    ["Target variables", ", ".join(project_summary["dataset_summary"]["target_variables"])],
                    ["Missing values total", project_summary["dataset_summary"]["missing_values_total"]],
                    ["Validation scheme", project_summary["dataset_summary"]["validation_scheme"]],
                    ["Train-test split ratio", project_summary["dataset_summary"]["train_test_split_ratio"]],
                ],
                columns=["Metric", "Value"]
            )
            st.dataframe(dataset_summary_df, use_container_width=True)

            st.subheader("Model Performance")
            st.dataframe(project_summary["model_performance_df"], use_container_width=True)

            if not project_summary["optimization_summary"]["forward_df"].empty:
                st.subheader("Forward Optimization")
                st.dataframe(project_summary["optimization_summary"]["forward_df"], use_container_width=True)

            if not project_summary["optimization_summary"]["reverse_df"].empty:
                st.subheader("Reverse Optimization")
                st.dataframe(project_summary["optimization_summary"]["reverse_df"], use_container_width=True)

            if not project_summary["feature_importance_df"].empty:
                st.subheader("Feature Importance Ranking")
                st.dataframe(project_summary["feature_importance_df"], use_container_width=True)

            st.subheader("Prediction Error Statistics")
            st.dataframe(project_summary["prediction_summary"]["prediction_metrics_df"], use_container_width=True)

            st.subheader("Concise Engineering Insights")
            for bullet in project_summary["observations"][:8]:
                st.write(f"- {bullet}")

            st.subheader("PPT-Ready Findings")
            for bullet in project_summary["ppt_ready_findings"]:
                st.write(f"- {bullet}")

            st.subheader("Results & Discussion")
            for bullet in project_summary["results_discussion"]:
                st.write(f"- {bullet}")

            st.subheader("Conclusion")
            for bullet in project_summary["conclusions"]:
                st.write(f"- {bullet}")

            st.subheader("Future Scope")
            for bullet in project_summary["future_scope"]:
                st.write(f"- {bullet}")

            st.subheader("Visualization Insights")
            for bullet in project_summary["visualization_insights"]:
                st.write(f"- {bullet}")

            st.download_button(
                "Download TXT",
                summary_to_text(project_summary),
                file_name="project_summary.txt",
                mime="text/plain"
            )

            st.download_button(
                "Download CSV",
                summary_to_flat_csv_df(project_summary).to_csv(index=False),
                file_name="project_summary.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download JSON",
                summary_to_json_str(project_summary),
                file_name="project_summary.json",
                mime="application/json"
            )

            docx_bytes = summary_to_docx_bytes(project_summary)
            if docx_bytes is not None:
                st.download_button(
                    "Download DOCX",
                    docx_bytes,
                    file_name="project_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.info("DOCX export requires the optional python-docx package.")
        else:
            st.info("Click Generate Project Summary after the dataset and optimization outputs are available.")