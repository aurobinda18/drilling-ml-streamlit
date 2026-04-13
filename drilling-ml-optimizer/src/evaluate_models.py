from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, X, y):
    """
    Evaluate model using Leave-One-Out Cross Validation (LOOCV)

    Returns:
    MAE
    RMSE
    R² score
    """

    loo = LeaveOneOut()

    predictions = []
    actuals = []

    for train_index, test_index in loo.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        predictions.append(pred[0])
        actuals.append(y_test.values[0])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    return mae, rmse, r2