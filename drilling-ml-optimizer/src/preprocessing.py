def split_features_targets(df):
    """
    Automatically separate input features (X)
    and output targets (y).

    Works whether dataset contains:
    Ra only
    Fd only
    or both.
    """

    # Possible prediction targets
    possible_targets = ["Ra", "Fd"]

    # Detect which targets exist
    targets = [col for col in possible_targets if col in df.columns]

    # Remove targets from feature set
    X = df.drop(columns=targets)

    # Remove Material column (text column not used directly)
    if "Material" in X.columns:
        X = X.drop(columns=["Material"])

    # Target dataframe
    y = df[targets]

    return X, y, targets