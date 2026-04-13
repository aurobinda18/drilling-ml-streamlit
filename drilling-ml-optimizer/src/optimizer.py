import pandas as pd
import numpy as np


def generate_parameter_grid(speed_range, feed_range, diameter,
                            youngs_modulus,
                            max_stress,
                            flex_modulus):
    """
    Create combinations of Speed and Feed values
    with fixed material properties.
    """

    grid = []

    for speed in speed_range:
        for feed in feed_range:

            grid.append([
                speed,
                feed,
                diameter,
                youngs_modulus,
                max_stress,
                flex_modulus
            ])

    columns = [
        "Speed",
        "Feed",
        "Diameter",
        "Modulus (Automatic Young's) (MPa)",
        "Maximum Stress (MPa)",
        "Flex Modulus (MPa)"
    ]

    return pd.DataFrame(grid, columns=columns)


def find_optimal_parameters(model, param_grid, targets):
    """
    Predict outputs for all parameter combinations
    and find minimum values.
    """

    predictions = model.predict(param_grid)

    results = param_grid.copy()

    if len(targets) == 1:
        results[targets[0]] = predictions

    else:
        for i, target in enumerate(targets):
            results[target] = predictions[:, i]

    optimal_rows = {}

    for target in targets:
        optimal_rows[target] = results.loc[results[target].idxmin()]

    return optimal_rows, results




def match_target_quality(model,
                         param_grid,
                         targets,
                         target_ra=None,
                         target_fd=None):
    """
    Find Speed and Feed settings that best match
    desired Ra and/or Fd values.
    Supports:
    - Ra only
    - Fd only
    - both together
    """

    predictions = model.predict(param_grid)

    results = param_grid.copy()

    # Attach predictions
    if len(targets) == 1:
        results[targets[0]] = predictions
    else:
        for i, target in enumerate(targets):
            results[target] = predictions[:, i]

    # Validate inputs
    if target_ra is None and target_fd is None:
        raise ValueError(
            "At least one target (Ra or Fd) must be provided."
        )

    # Compute matching error
    if target_ra is not None and "Ra" in targets:
        results["ra_error"] = abs(results["Ra"] - target_ra)
    else:
        results["ra_error"] = 0

    if target_fd is not None and "Fd" in targets:
        results["fd_error"] = abs(results["Fd"] - target_fd)
    else:
        results["fd_error"] = 0

    results["match_error"] = results["ra_error"] + results["fd_error"]

    best_match = results.loc[results["match_error"].idxmin()]

    return best_match