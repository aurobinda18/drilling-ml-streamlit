def split_features_targets(df):

    """
    Automatically detect targets and prepare feature matrix.
    Converts categorical variables to numeric using one-hot encoding.
    """

    import pandas as pd

    # Common machining output keywords
    possible_targets = [
        "Ra",
        "Fd",
        "Surface_Roughness",
        "Delamination",
        "Force",
        "Temperature",
        "Wear"
    ]

    detected_targets = []

    for col in df.columns:

        if any(keyword.lower() in col.lower()
               for keyword in possible_targets):

            detected_targets.append(col)

    # fallback if none detected
    if len(detected_targets) == 0:

        numeric_cols = df.select_dtypes(include="number").columns

        detected_targets = numeric_cols[-1:]

    # Normalize target columns to numeric where possible. This avoids
    # downstream model/plotting errors when CSV decimals are parsed as text.
    for target_col in detected_targets:
        if target_col in df.columns:
            cleaned = pd.to_numeric(
                df[target_col].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce"
            )

            if cleaned.notna().sum() == 0:
                raise ValueError(
                    f"Target column '{target_col}' could not be parsed as numeric values."
                )

            df[target_col] = cleaned

    # Split features and targets
    X = df.drop(columns=detected_targets)
    y = df[detected_targets]

    # Convert categorical features into numeric
    X = pd.get_dummies(X, drop_first=True)

    return X, y, detected_targets