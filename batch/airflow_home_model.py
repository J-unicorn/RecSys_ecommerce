from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.models import XCom
import yaml
from models.item2vec_model import  * 
from models.topn_model import  * 
from models.lgcn_model import  * 
from models.ensemble import  * 
from AgensConnector import *
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
connect_info = config["connect_info"]
ag_session = AgensConnector(**connect_info)

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 2, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    '_interval': '*/30 * * * *',
}

# # 병렬 CPU 사용을 위한 executor_config 설정
# executor_config = {
#     "KubernetesExecutor": {
#         "request_cpus": "1",
#     }
# }


# The main DAG
main_dag = DAG(
    dag_id='Home_RecSys',
    default_args=default_args,
    catchup=False,
)

# Initialize the TopN class
topn = TopN(ag_session)
item2vec = Item2Vec(ag_session)
lgcn =  Lightgcn(ag_session)
ensemble = Ensemble(ag_session)

models = [item2vec, topn, lgcn]


# 각 작업을 처리하는 함수 코드 남김

# Each task function

def preprocess_data(model, execution_date,  **context):
    end_date = datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=127)
    start_date = end_date - timedelta(days=7)
    view_cnt = 5 if type(model).__name__== 'Item2Vec' else 15 
    print(view_cnt)

    #output = model.preprocess_data(start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'), view_cnt)
    output = model.preprocess_data('2023-05-24','2023-06-01', view_cnt)
    
    # XCom push
    push_key = type(model).__name__ + 'preprocessed_data'
    context['ti'].xcom_push(key=push_key, value=output)
    return output

def save_model(model, **context):
    
    
    # XCom pull
    pull_key = type(model).__name__ + 'preprocessed_data'
    input_data =context['ti'].xcom_pull(key=pull_key)
    print(model,input_data)
    
    output = model.saved_model(*input_data)
    # XCom push
    push_key = type(model).__name__ + 'saved_model'
    context['ti'].xcom_push(key=push_key, value=output)
    
    
    return output

def setup_ensemble(model,ensemble, **context):
    
    
    # XCom pull
    pull_key = type(model).__name__ + 'saved_model'
    input_data =context['ti'].xcom_pull(key=pull_key)
    
    
    print(model,input_data)
    
    
    output = ensemble.get_padding_score(input_data)
    
    return output

def predict(ensemble,weights_list):
        
    output = ensemble.get_weighted_ensemble(weights_list)

    
    return output



with main_dag:
    with TaskGroup('train_models') as train_models:
        for model in models:
            model_name=type(model).__name__
            with TaskGroup(f'{model_name}_tasks') as model_tasks:
         

                preprocess_task = PythonOperator(
                    task_id=f'preprocess_data_{model_name}',
                    python_callable=preprocess_data,
                    op_kwargs={'model': model, 'execution_date': '{{ ds }}' },
                    provide_context=True,
                    dag=main_dag,
                   # executor_config=executor_config, # 병렬 CPU 사용 설정 추가
                )

                save_task = PythonOperator(
                    task_id=f'save_model_{model_name}',
                    python_callable=save_model,
                    op_kwargs={'model': model},
                    provide_context=True,
                    dag=main_dag,
                    #executor_config=executor_config, # 병렬 CPU 사용 설정 추가
                )

                setup_ensemble_task = PythonOperator(
                    task_id=f'setup_ensemble_{model_name}',
                    python_callable=setup_ensemble,
                    op_kwargs={'model': model,'ensemble':ensemble},
                    provide_context=True,
                    dag=main_dag,
                    #executor_config=executor_config, # 병렬 CPU 사용 설정 추가
                )

                # 모델 DAG 내에서 순차적으로 작업되도록 설정
                preprocess_task >> save_task >> setup_ensemble_task


    predict_task = BashOperator(
        task_id='predict_api',
        bash_command = '/home/airflow/airflow/dags/streamlit/start_fastapi.sh',
        dag = main_dag,
    )
    # predict_task = PythonOperator(
    #     task_id='predict_task',
    #     python_callable=predict,
    #     op_kwargs={'ensemble':ensemble,'weights_list':[0.4,0.2,0.4]},              
    #     provide_context=True,
    #     dag=main_dag,
    # )

    # taskgroup을 사용해 각 train model 작업 병렬 수행 후 predict 작업 수행 
    train_models >> predict_task
