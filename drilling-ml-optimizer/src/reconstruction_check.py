from sklearn.metrics import mean_squared_error
import numpy as np


def reconstruction_test(model, X, y):
    """
    Check how well model reproduces original dataset outputs.

    Lower error = better reconstruction accuracy
    """

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)

    return rmse