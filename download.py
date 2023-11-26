from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import zipfile
import os

# Функция для распаковки архива
def extract_archive(archive_name):
    with zipfile.ZipFile(archive_name, 'r') as zip_ref:
        zip_ref.extractall('.')  # Распаковка архива в текущую директорию

# Функция для удаления файла архива
def delete_archive(archive_name):
    os.remove(archive_name)  # Удаление файла архива

def download_kaggle_dataset(dataset_name):
    command = f'kaggle datasets download -d {dataset_name}'
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Датасет {dataset_name} успешно загружен.")

        # Распаковка архива
        archive_name = f"{dataset_name.split('/')[-1]}.zip"  # Определение имени архива
        extract_archive(archive_name)

        # Удаление архива после распаковки
        delete_archive(archive_name)
        print(f"Архив {archive_name} успешно удален.")
    except subprocess.CalledProcessError as e:
        print(f"Произошла ошибка при загрузке датасета: {str(e)}")

if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    dataset_name = 'quantbruce/real-estate-price-prediction'  # Укажите название датасета
    download_kaggle_dataset(dataset_name)
