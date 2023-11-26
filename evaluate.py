import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Загрузка данных для тестирования модели
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    # Загрузка обученной модели
    trained_model = xgb.XGBRegressor()
    trained_model.load_model('trained_xgb_model.model')

    # Прогнозирование на тестовом наборе данных
    y_pred = trained_model.predict(X_test)

    # Оценка качества модели
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE модели на тестовых данных: {rmse}")
