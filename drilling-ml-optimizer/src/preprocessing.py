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

    # Split features and targets
    X = df.drop(columns=detected_targets)
    y = df[detected_targets]

    # Convert categorical features into numeric
    X = pd.get_dummies(X, drop_first=True)

    return X, y, detected_targets