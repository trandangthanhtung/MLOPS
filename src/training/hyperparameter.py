from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_model_configs():
    """
    Return config for all models in pipeline
    """

    return {
        # =========================
        # LSTM (TensorFlow)
        # =========================
        "LSTM": {
            "type": "deep_learning",
            "params": {
                "units": 128,
                "dropout": 0.3,
                "learning_rate": 0.001,
                "epochs": 30,
                "batch_size": 64
            }
        },

        # =========================
        # Random Forest
        # =========================
        "RandomForest": {
            "type": "sklearn",
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "n_jobs": -1
            }
        },

        # =========================
        # XGBoost
        # =========================
        "XGBoost": {
            "type": "sklearn",
            "model": XGBRegressor,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42
            }
        }
    }