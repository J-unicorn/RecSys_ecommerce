from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.models import XCom
from models.itemknn_model import  * 
from models.sessiongraph_model import  * 
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
    '_interval': '*/10 * * * *',
}

# The main DAG
main_dag = DAG(
    dag_id='Item_RecSys',
    default_args=default_args,
    catchup=False,
)

# Initialize the TopN class
itemknn = ItemKNN(ag_session)
sessiongraph = SessionGraph(ag_session)

models = [itemknn, sessiongraph]


# 각 작업을 처리하는 함수 코드 남김

# Each task function

def save_model(model, execution_date,  **context):
    end_date = datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=127)
    #start_date =  end_date - timedelta(months=1) if type(model).__name__== 'Item2Vec' else end_date - timedelta(days=1)
    start_date =  end_date - timedelta(days=1)
 
   # output = model.preprocess_data(start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S'), view_cnt)
    model.saved_model('2023-05-31 23:00:00','2023-06-01')
    
    return 



with main_dag:
    itemknn_task = PythonOperator(
                    task_id=f'save_model_itemknn',
                    python_callable=save_model,
                    op_kwargs={'model': itemknn, 'execution_date': '{{ ds }}' },
                    provide_context=True,
                    dag=main_dag,
                    #executor_config=executor_config, # 병렬 CPU 사용 설정 추가
                )
    sessiongraph_task = PythonOperator(
                    task_id=f'save_model_sessiongraph',
                    python_callable=save_model,
                    op_kwargs={'model': sessiongraph, 'execution_date': '{{ ds }}' },
                    provide_context=True,
                    dag=main_dag,
                    #executor_config=executor_config, # 병렬 CPU 사용 설정 추가
                )
  
    # with TaskGroup('train_models') as train_models:
    #     for model in models:
    #         model_name=type(model).__name__
    #         with TaskGroup(f'{model_name}_tasks') as model_tasks:
            
    #             save_task = PythonOperator(
    #                 task_id=f'save_model_{model_name}',
    #                 python_callable=save_model,
    #                 op_kwargs={'model': model, 'execution_date': '{{ ds }}' },
    #                 provide_context=True,
    #                 dag=main_dag,
    #                 #executor_config=executor_config, # 병렬 CPU 사용 설정 추가
    #             )

                

    #             # 모델 DAG 내에서 순차적으로 작업되도록 설정
    #             save_task 


    predict_task = BashOperator(
        task_id='predict_api',
        bash_command = '/home/airflow/airflow/dags/streamlit/start_fastapi.sh',
        dag = main_dag,
    )

    # taskgroup을 사용해 각 train model 작업 병렬 수행 후 predict 작업 수행 
    itemknn_task >> sessiongraph_task >> predict_task
