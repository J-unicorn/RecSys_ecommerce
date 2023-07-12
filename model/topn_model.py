import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List
import os
import pickle as pkl
from .agquery import *
#from agquery import *
import time

#import re
#import multiprocessing
#cores = multiprocessing.cpu_count()

#import os
# 병렬 처리에 사용할 스레드 수 설정
#os.environ['OMP_NUM_THREADS'] = str(cores)

class TopN:
    def __init__(self, session):
        self.session = session
        self.path = '/home/airflow/airflow/dags/models/topn'    
            
    
    
    def preprocess_data(self, start_date, end_date, view_cnt):
        
        # AG에서 데이터를 로드
        df_category=self.session.query_pandas(category_stat_query(start_date,end_date,view_cnt))
        
        df_view=self.session.query_pandas(member_view_query(start_date,end_date))
        #df_view=df_view[df_view.member_idx.isin(df_view.member_idx.value_counts().loc[lambda x:x >= 5].index)].reset_index(drop=True)

        # 데이터 전처리 및 피처 추출 등의 작업 수행

        pivot_user = pd.pivot_table(df_view, index='member_idx', columns='param_category_id', values='scale', fill_value=0)
        pivot_item = pd.pivot_table(df_category, index='param_category_id', columns='item_idx', values='scale', fill_value=0)
        diff_col=np.setdiff1d(pivot_user.columns,pivot_item.index)    
        if len(diff_col) > 0:
            for i in diff_col:
                pivot_user.drop(columns=i,  inplace=True)
        user_dict = {k:i for i,k in enumerate(pivot_user.index)}
        item_dict = {k:i for i,k in enumerate(pivot_item.columns)}
        user_dict_path = os.path.join(self.path,'user_dict.pkl')
        item_dict_path = os.path.join(self.path,'item_dict.pkl')
        
        with open(file=user_dict_path,mode='wb') as f:
            pkl.dump(user_dict,f)
        with open(file=item_dict_path,mode='wb') as f:
            pkl.dump(item_dict,f)
        
        
        user_cat_mat = csr_matrix(pivot_user.values)
        cat_item_mat = csr_matrix(pivot_item.values)

        user_cat_mat_path = os.path.join(self.path,'user_cat_mat.pkl')
        cat_item_mat_path = os.path.join(self.path,'cat_item_mat.pkl')

        #np.save(user_cat_mat_path,user_cat_mat)
        #np.save(cat_item_mat_path,cat_item_mat)
        with open(file=user_cat_mat_path,mode='wb') as f:
            pkl.dump(user_cat_mat,f)
        with open(file=cat_item_mat_path,mode='wb') as f:
            pkl.dump(cat_item_mat,f)

       # print('user_category matrix : ',user_cat_mat.shape)
       # print('category_item matrix : ',cat_item_mat.shape)
        
        return user_cat_mat_path,cat_item_mat_path
    
    def saved_model(self,  user_cat_mat_path,cat_item_mat_path):

        #user_cat_mat = np.load(user_cat_mat_path + '.npy')
        #cat_item_mat = np.load(cat_item_mat_path + '.npy')
        with open(user_cat_mat_path, "rb") as f:
            user_cat_mat = pkl.load(f)
        with open(cat_item_mat_path, "rb") as f:
            cat_item_mat = pkl.load(f)
        
        
        user_item_mat =np.dot(user_cat_mat, cat_item_mat)
        user_item_mat = user_item_mat.toarray()
    
        topn_mat_path = os.path.join(self.path,'topn_mat')    
        np.save(topn_mat_path,user_item_mat)
        
        #with open(file=topn_mat_path,mode='wb') as f:
        #    pkl.dump(user_item_mat,f)
        
        '''
        predicted_matrix = np.dot(user_seen_mat, item_sim_mat)
        already_seen = np.isnan(user_seen_mat.toarray())
        predicted_matrix[already_seen] = -np.inf
        predicted_matrix = predicted_matrix.toarray()
        '''

            
        return topn_mat_path
    
    
    def train(self, start_date, end_date, cnt):
        user_cat_mat,cat_item_mat = self.preprocess_data(start_date, end_date, cnt)
        predicted_matrix=self.saved_model(user_cat_mat,cat_item_mat)
