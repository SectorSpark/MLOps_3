import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(file_path):
    # Загрузка данных из файла CSV
    data = pd.read_csv(file_path)

    # Удаление столбца 'No', так как это, похоже, идентификатор строки, который не представляет ценности для модели
    data.drop('No', axis=1, inplace=True)

    # Преобразование X1 transaction date в числовой формат (может потребоваться дополнительная обработка для этого столбца)
    data['X1 transaction date'] = pd.to_numeric(data['X1 transaction date'], errors='coerce')

    # Пример масштабирования числовых признаков (в данном случае, масштабирование всех числовых признаков)
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

def save_train_test_data(X_train, X_test, y_train, y_test):
    os.makedirs('data', exist_ok=True)

    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

if __name__ == "__main__":
    file_path = 'Real estate.csv'  # Укажите путь к вашему CSV файлу
    preprocessed_data = preprocess_data(file_path)

    # Разделение данных на признаки (X) и целевую переменную (y)
    X = preprocessed_data.drop('Y house price of unit area', axis=1)
    y = preprocessed_data['Y house price of unit area']

    # Разделение на обучающий и тестовый наборы данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Сохранение данных в папку 'data'
    save_train_test_data(X_train, X_test, y_train, y_test)

    print("Данные для обучения и тестирования сохранены в папке 'data'.")
