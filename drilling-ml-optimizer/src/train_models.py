from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_models(X, y):
    """
    Train multiple regression models on dataset.

    Supports both:
    single-target prediction
    multi-target prediction (Ra + Fd)
    """

    models = {}

    # Model 1: Linear Regression
    models["Linear Regression"] = LinearRegression()

    # Model 2: Random Forest
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Model 3: Gradient Boosting (wrapped for multi-output support)
    models["Gradient Boosting"] = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    )

    trained_models = {}

    # Train each model
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model

    return trained_models