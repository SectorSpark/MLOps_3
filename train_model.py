import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split



def train_xgb_model(X_train, y_train):
    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("colsample_bytree", 0.3)
        mlflow.log_param("learning_rate", 0.15)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("alpha", 10)
        mlflow.log_param("n_estimators", 400)

        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.4, learning_rate=0.05,
                                 max_depth=8, alpha=10, n_estimators=400)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        mlflow.sklearn.log_model(model, "xgb_model")

if __name__ == "__main__":
    # Load training data
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    train_xgb_model(X_train, y_train)
