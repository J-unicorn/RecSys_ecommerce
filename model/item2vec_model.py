import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from typing import List
import pickle as pkl
import re
import multiprocessing
cores = multiprocessing.cpu_count()

from .agquery import *
#from agquery import *


import os
# 병렬 처리에 사용할 스레드 수 설정
os.environ['OMP_NUM_THREADS'] = str(cores)

class Item2Vec:
    def __init__(self, session):
        self.session = session
        self.path = '/home/airflow/airflow/dags/models/item2vec'
        self.knnpath = '/home/airflow/airflow/dags/models/itemknn'
    
    def preprocess_data(self, start_date, end_date, cnt):
               
         # AG에서 데이터를 로드

        
        user_view = self.session.query_pandas(view_query_item2vec(start_date, end_date))
        user_view=user_view[user_view.item_idx.isin(user_view.item_idx.value_counts().loc[lambda x:x >= cnt].index)].reset_index(drop=True)

        num_user = user_view['member_idx'].nunique()
        num_item = user_view['item_idx'].nunique()

        print('# of user :' ,num_user)
        print('# of item :' ,num_item)

        user_dict = {k:i for i, k in enumerate(user_view.member_idx.unique())}
        item_dict = {k:i for i, k in enumerate(user_view.item_idx.unique())}

        
        user_dict_path = os.path.join(self.path,'user_dict.pkl')
        item_dict_path = os.path.join(self.path,'item_dict.pkl')
        
        with open(file=user_dict_path,mode='wb') as f:
            pkl.dump(user_dict,f)
        
        with open(self.knnpath+'/rev_itemknn_dict.pkl', 'rb') as f:
           rev_itemknn_dict = pkl.load(f)
        itemknn_dict = {v:k for k,v in rev_itemknn_dict.items()}

        itemknn_score = np.load(self.knnpath+'/itemknn_score.npy')
        itemknn_index = np.load(self.knnpath+'/itemknn_index.npy')
        print('index : ',itemknn_index.shape)
        print('score : ',itemknn_score.shape)

        item_list=list(item_dict.keys())
        item_num = len(item_list)
        itemknn_idx = np.array([itemknn_dict[i] for i in item_list])
        itemknn_index1,itemknn_score1=itemknn_index[itemknn_idx],itemknn_score[itemknn_idx]
      #  del itemknn_score
       # del itemknn_index
        target_item =np.unique(itemknn_index1.flatten())
        target_num=target_item.shape[0]
        predict_dict={rev_itemknn_dict[k]:i for i,k in enumerate(target_item)}
        
        with open(file=item_dict_path,mode='wb') as f:
            pkl.dump(predict_dict,f)
        
        
        itemknn_mat = np.zeros((item_num,target_num))
        mat_dict =  {k:i for i,k in enumerate(np.unique(target_item))}
        converted_index = np.array([mat_dict.get(i) for i in itemknn_index1.flatten()]).reshape(itemknn_index1.shape)
       # del itemknn_index1
        itemknn_mat[np.arange(item_num)[:, None], converted_index] = itemknn_score1.reshape(item_num, -1)
        itemknn_matrix = csr_matrix(itemknn_mat)
        
        #inv_item_dict = {i:k for i, k in enumerate(user_view.item_idx.unique())}
        #user_view = np.array()
        user_view_path = os.path.join(self.path,'user_view.pkl')
        itemknn_matrix_path = os.path.join(self.path,'itemknn_matrix.pkl')
        
        with open(file=user_view_path,mode='wb') as f:
            pkl.dump(user_view,f)
        with open(file=itemknn_matrix_path,mode='wb') as f:
            pkl.dump(itemknn_matrix,f)

        
        #np.save(user_view_path,user_view)
    #     with open(file=user_seen_mat_path,mode='wb') as f:
    #         pkl.dump(user_seen_mat,f)
        
        return user_view_path,itemknn_matrix_path
        
    #def saved_model(self, user_view_path,num_user,num_item ):        
    def saved_model(self, user_view_path,itemknn_matrix_path ):        
         

        with open(itemknn_matrix_path, 'rb') as f:
            itemknn_matrix = pkl.load(f) 
        
        #user_view_path =  self.path + '/user_view.pkl'
        user_view =pd.read_pickle(user_view_path)
        #user_view=np.load(user_view_path+'.npy')
        #user_view=pd.DataFrame(user_view,columns=['member_idx','item_idx','time_w'])
        
        num_user = user_view['member_idx'].nunique()
        num_item = user_view['item_idx'].nunique()
        
        user_dict = {k:i for i, k in enumerate(user_view.member_idx.unique())}
        item_dict = {k:i for i, k in enumerate(user_view.item_idx.unique())}

        
        #for i in tqdm(range(item_num)):
        #   itemknn_mat[i, converted_index[i]] = itemknn_score1[i]
        interaction_matrix = np.zeros((num_user,num_item))
        for value in user_view.values:
            item_idx = item_dict[value[1]]
            member_idx = user_dict[value[0]]
            interaction_matrix[member_idx, item_idx] = value[2]
        interaction_matrix = csr_matrix(interaction_matrix)
        itemknn_predict = np.dot(interaction_matrix,itemknn_matrix)
       # del interaction_matrix
       # del itemknn_matrix
        itemknn_predict=itemknn_predict.toarray()
        
        model_scores = np.zeros_like(itemknn_predict)  # 결과 배열 초기화

        scaler = preprocessing.MinMaxScaler()  # scaler 인스턴스 생성
        
        for i in range(itemknn_predict.shape[0]):
            row = itemknn_predict[i]
            nonzero_indices = row.nonzero()[0]  # 0이 아닌 값의 인덱스를 선택
            nonzero_values = row[nonzero_indices]  # 0이 아닌 값들을 선택
            transformed_values = scaler.fit_transform(nonzero_values.reshape(-1, 1))  # 0이 아닌 값에 대해 MinMaxScaler 적용
            row[nonzero_indices] = transformed_values.flatten()  # 변환된 값을 해당 위치에 할당
            model_scores[i] = row  # 변환된 row를 model_scores에 할당

        predict_matrix_path = os.path.join(self.path,'item2vec_mat')
        
        np.save(predict_matrix_path,model_scores)
        # with open(file=predict_matrix_path,mode='wb') as f:
        #     pkl.dump(predicted_matrix,f)
        print('item2vec_mat_success')    
        return predict_matrix_path
      
    
    def train(self, start_date, end_date, cnt):
       
        item_meta,user_seen_mat = self.preprocess_data(start_date, end_date, cnt)
        predicted_matrix=self.saved_model(item_meta,user_seen_mat)