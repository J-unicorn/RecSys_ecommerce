import numpy as np
import pandas as pd
import os
import time
import pickle as pkl
import torch
from torch import optim
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
from .lightgcn.model import LightGCN, BPR_loss
from .lightgcn.metrics import *
from .lightgcn.utils import *
from .agquery import *

import warnings; warnings.filterwarnings('ignore')
from typing import List
import pickle as pkl

class Args:
    def __init__(self):
        self.epochs = 1
        self.check_step = 10
        self.batch_size = 2**15
        self.lr = 0.05
        self.lambda_ = 1e-3
        self.test_size = 0.15
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = 1000
        self.embedding_dim = 300
        self.num_layers = 3
        self.saved_path = '../saved/'
        self.eval_mode = 'online'
        self.insert = False
        

class Lightgcn:
    def __init__(self, session):
        self.session = session
        self.args = Args()
        self.path = '/home/airflow/airflow/dags/models/lightgcn'
           
   
    def preprocess_data(self, start_date, end_date, cnt):
        # AG에서 데이터를 로드
        load_view = self.session.query_pandas(view_query_lgcn(start_date, end_date, cnt))
        load_view=load_view[load_view.member_idx.isin(load_view.member_idx.value_counts().loc[lambda x:x >= 5].index)].reset_index(drop=True)

        array = np.array(load_view)
        u,v = array[:,0] , array[:,1]
        user_mapping = {index: i for i, index in enumerate(np.unique(u))}
        item_mapping = {index: i for i, index in enumerate(np.unique(v))}
        src = torch.tensor([user_mapping[i] for i in u])
        dst = torch.tensor([item_mapping[i] for i in v])
        edge_index = torch.stack((src, dst))
        
        
        user_mapping_path = os.path.join(self.path,'user_dict.pkl')
        item_mapping_path = os.path.join(self.path,'item_dict.pkl')
        edge_index_path = os.path.join(self.path,'edge_index.pkl')
        with open(user_mapping_path, 'wb') as f:
            pkl.dump(user_mapping, f)
        with open(item_mapping_path, 'wb') as f:
            pkl.dump(item_mapping, f)
        with open(edge_index_path, 'wb') as f:
            pkl.dump(edge_index, f)
        
        
        return user_mapping_path, item_mapping_path, edge_index_path 

    
    def saved_model(self,  user_mapping_path, item_mapping_path, edge_index_path ):

      #  data = np.array(pd.DataFrame(load_view))
        args =  self.args
        with open(user_mapping_path, "rb") as f:
            user_mapping = pkl.load(f)
        with open(item_mapping_path, "rb") as f:
            item_mapping = pkl.load(f)
        with open(edge_index_path, "rb") as f:
            edge_index = pkl.load(f) 
       # u,v = data[:,0] , data[:,1]

        num_users, num_items = len(user_mapping), len(item_mapping)
     
        train_edge_index = edge_index
        train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1]+num_users, 
                                       sparse_sizes=(num_users + num_items, num_users + num_items))
        edge_indexes = train_edge_index, train_sparse_edge_index,
        
        
        print(f'num users in train data: {num_users} \nnum items in train data: {num_items}')
        model = LightGCN(num_users, num_items, args.embedding_dim, args.num_layers)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_edge_index, train_sparse_edge_index = edge_indexes[0], edge_indexes[1]
        print('LightGCN training on whole dataset start.')
        max_score = 0
        elapsed = 0
        start = time.time()
        for epoch in tqdm(range(1, args.epochs+1)):

            model.train()
            trn_loader = DataLoader(train_edge_index.T, args.batch_size, shuffle=True)
            trn_loss = 0

            for batch_pos_edges in trn_loader:

                loss = BPR_loss(model, train_edge_index, train_sparse_edge_index, batch_pos_edges, args.lambda_, num_users, num_items, args.device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # optional

                trn_loss += loss.item()

            trn_loss = trn_loss / len(trn_loader)

            if epoch % args.check_step == 0:
                elapsed = time.time()-start
                print(f'[{epoch:03d}/{args.epochs:03d}] | loss: {trn_loss:.6f} | elapsed time: {elapsed:.2f}s')

        print(f'Training on whole dataset complete.')
        # model_saved_as = os.path.join(args.saved_path, 'model_params.pt')
        # torch.save(model.state_dict(), model_saved_as)
        # print(f'Trained model saved in {model_saved_as}.')
        scores = get_score_matrix(model)
        #scores = get_filtered_score_matrix(scores, data, user_mapping, item_mapping)
        score_saved_as = os.path.join(self.path,'lightgcn_mat')
        np.save(score_saved_as,scores)
        # with open(score_saved_as, 'wb') as f:
        #     pkl.dump(scores, f)
        
        print(f'Score matrix saved in {score_saved_as}. (shape: {scores.shape})') #scores
    

        #final_elapsed = np.round(time.time()-start, 0)
        return score_saved_as
        
    
    def train(self, start_date, end_date, cnt):
       
        edge_indexes, num_users, num_items, user_mapping, item_mapping = self.preprocess_data(start_date, end_date, cnt)
        predicted_matrix=self.saved_model(edge_indexes, num_users, num_items, user_mapping, item_mapping)