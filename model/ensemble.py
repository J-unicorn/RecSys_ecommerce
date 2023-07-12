import numpy as np
import pandas as pd
import os
import time
import pickle as pkl
from tqdm import tqdm
import json
from tempfile import mkdtemp
from sklearn import preprocessing
from joblib import Parallel, delayed


import warnings; warnings.filterwarnings('ignore')
from typing import List


class Ensemble:
    def __init__(self, session):
        self.session = session
        self.path = '/home/airflow/airflow/dags/models/ensemble'
        self.parent_path = '/home/airflow/airflow/dags/models'
        self.models_name = ['topn', 'item2vec', 'lightgcn']
        self.batch_size = 10
           
    def get_useritem_lst(self,model,dict=False):

        user_dict_path = os.path.join(self.parent_path,f'{model}/user_dict.pkl')
        item_dict_path = os.path.join(self.parent_path,f'{model}/item_dict.pkl')
        
        with open(user_dict_path, 'rb') as f:
            model_user_dict = pkl.load(f)
        with open(item_dict_path, 'rb') as f:
            model_item_dict = pkl.load(f)
        model_ulst = np.array(list(model_user_dict.keys()))
        model_ilst = np.array(list(model_item_dict.keys()))
        
        if not dict:
            return model_ulst,model_ilst
        else:
            return model_ulst,model_ilst,model_user_dict,model_item_dict 
            
    def get_total_union(self):

        model1,model2,model3 = self.models_name
        model1_ulst,model1_ilst = self.get_useritem_lst(model1)
        model2_ulst,model2_ilst = self.get_useritem_lst(model2)
        model3_ulst,model3_ilst = self.get_useritem_lst(model3)
        user_union = np.union1d(np.union1d(model1_ulst, model2_ulst), model3_ulst)
        item_union = np.union1d(np.union1d(model1_ilst, model2_ilst), model3_ilst)
        total_users = len(user_union)
        total_items = len(item_union)
        tot_user_dict = {k:i for i,k in enumerate(user_union)}
        tot_item_dict = {k:i for i,k in enumerate(item_union)}
        tot_user_dict_path = os.path.join(self.path,'tot_user_dict.pkl')
        tot_item_dict_path = os.path.join(self.path,'tot_item_dict.pkl')

        
        with open(tot_user_dict_path, 'wb') as f:
            pkl.dump(tot_user_dict, f)
        with open(tot_item_dict_path, 'wb') as f:
            pkl.dump(tot_item_dict, f)
        
        return total_users,total_items,tot_user_dict,tot_item_dict
     
    def get_padding_score(self,model_mat_path,scaling=False):

        model = model_mat_path.split('/')[-2]
        total_users,total_items,tot_user_dict,tot_item_dict= self.get_total_union()
        model_ulst,model_ilst,model_user_dict,model_item_dict = self.get_useritem_lst(model,dict=True)
        model_aligned = np.zeros((total_users, total_items))

        #model_mat_path = f'{model}/{model}_mat.pkl'
        model_mat=np.load(model_mat_path+'.npy')        
        # with open(model_mat_path, 'rb') as f:
        #     model_mat = pkl.load(f)

   
        # 벡터화된 연산을 위해 인덱스 매핑
        model_user_idx = np.array([model_user_dict[j] for j in model_ulst])
        model_item_idx = np.array([model_item_dict[i] for i in model_ilst])
        aligned_user_idx = np.array([tot_user_dict[j] for j in model_ulst])
        aligned_item_idx = np.array([tot_item_dict[i] for i in model_ilst])
        
        # 새로운 score matrix를 위한 행렬 초기화
        model_aligned = np.zeros((total_users, total_items))
        
        # 벡터화된 연산을 활용한 score matrix 매핑
        model_aligned[np.ix_(aligned_user_idx, aligned_item_idx)] = model_mat[np.ix_(model_user_idx, model_item_idx)]
        # parallel_map_scores()를 사용하여 병렬 처리
        
        if scaling == True: 
            model_scaler = preprocessing.MinMaxScaler()
            model_scores = model_scaler.fit(model_aligned).transform(model_aligned)
        else:
            model_scores = model_aligned

        model_score_path = self.make_path(f'{model}_aligned_score', directory=self.path)
        new_np           = self.make_memmap(model_score_path , model_scores)
        # with open(model_score_path, 'wb') as f:
        #     pkl.dump(model_scores, f)
        del model_scores, new_np
        return model_score_path      
    
    def make_memmap(self,mem_file_name, np_to_copy):
        # numpy.ndarray객체를 이용하여 numpy.memmap객체를 만든다
        memmap_configs = dict() # memmap config 저장할 dict
        memmap_configs['shape'] = shape = tuple(np_to_copy.shape) # 형상 정보
        memmap_configs['dtype'] = dtype = str(np_to_copy.dtype)   # dtype 정보
        json.dump(memmap_configs, open(mem_file_name+'.conf', 'w')) # 파일 저장
        # w+ mode: Create or overwrite existing file for reading and writing
        mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
        mm[:] = np_to_copy[:]
        mm.flush() # memmap data flush
        return mm
    
    def make_path(self,file_name, directory='', is_make_temp_dir=False):
        """디렉토리와 파일명을 더해 경로를 만든다"""
        if is_make_temp_dir is True:
            directory = mkdtemp()
        if len(directory) >= 2 and not os.path.exists(directory):
            os.makedirs(directory)    
        return os.path.join(directory, file_name)

    
    def pred_func(self,member_idx, item_dict, score_mat, topk=False):
        arr = score_mat[member_idx]
        res = (-arr).argsort()
        res = [ item_dict[i] for i in res]
        return res[:topk] if topk else res    

    
    def insert_data_batch(self,user_list, item_list, rank_list, batch_size):
        query = 'INSERT INTO ensembles_result (member_idx, item_idx, ranking_point) SELECT unnest(%(user)s), unnest(%(item)s), unnest(%(rank)s)'
        for i in tqdm(range(0, len(user_list), batch_size)):
            data = {'user': user_list[i:i+batch_size], 'item': item_list[i:i+batch_size], 'rank': rank_list[i:i+batch_size]}
            self.session.query_exe(query, data=data)
        return 
    
    def get_weighted_ensemble (self,weights_list):

        w1, w2, w3 = weights_list
        models_name = self.models_name
        model_path1 = os.path.join(self.path,f'{models_name[0]}_aligned_score.pkl')
        model_path2 = os.path.join(self.path,f'{models_name[1]}_aligned_score.pkl')
        model_path3= os.path.join(self.path,f'{models_name[2]}_aligned_score.pkl')
        
        with open(model_path1, 'rb') as f:
            model1_scores = pkl.load(f)
        with open(model_path2, 'rb') as f:
            model2_scores = pkl.load(f)
        with open(model_path3, 'rb') as f:
            model3_scores = pkl.load(f)
        
        weights_scores = w1*model1_scores + w2*model2_scores + w3*model3_scores

        weights_scores_path= os.path.join(self.path,'weights_score.pkl')
        tot_user_dict_path = os.path.join(self.path,'tot_user_dict.pkl')
        tot_item_dict_path = os.path.join(self.path,'tot_item_dict.pkl')
        
        with open(weights_scores_path, 'wb') as f:
            pkl.dump(weights_scores,f)
    
        with open(tot_user_dict_path, 'rb') as f:
            tot_user_dict = pkl.load(f)
        with open(tot_item_dict_path, 'rb') as f:
            tot_item_dict = pkl.load(f)
        
        inv_tot_user_dict = {v: k for k, v in tot_user_dict.items()}
        inv_tot_item_dict = {v: k for k, v in tot_item_dict.items()}
        pred_item = [int(item) for user in tot_user_dict.keys() for item in self.pred_func(user, inv_tot_item_dict, weights_scores, 1000)]
        #pred_item_list = [user for i in list(tot_user_dict.keys()) for self.pred_func(i, tot_item_dict, weights_scores, 1000)]
        pred_user_list = tot_user_dict.values()
        pred_user = [int(num) for num in pred_user_list for _ in range(1000)]
        pred_rank= [int(i) for i in  np.tile(np.arange(1,1001),len(pred_user_list))]

        # batch_size=len(pred_user)//self.batch_size
        # pred_user =[int(i) for i in pred_user]
        # pred_item =[int(i) for i in pred_item_list]
        # pred_rank =[int(i) for i in rank_list]
        
        #self.insert_data_batch(pred_user, pred_item, pred_rank, batch_size)
        
        return weights_scores_path

