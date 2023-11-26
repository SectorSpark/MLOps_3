from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 11, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('model_training_dag', default_args=default_args, schedule_interval='@daily')

# Task 1: Training the model
def train_model():
    import subprocess
    subprocess.run(["python3", "/path/to/train_model.py"])

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag
)

# Task 2: Checking the model
def check_model():
    import subprocess
    subprocess.run(["python3", "/path/to/evaluate.py"])

check_model_task = PythonOperator(
    task_id='check_model_task',
    python_callable=check_model,
    dag=dag
)

# Define task dependencies
train_model_task >> check_model_task
