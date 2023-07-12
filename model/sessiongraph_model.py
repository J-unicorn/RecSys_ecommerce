#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mmap
import time
from tqdm import tqdm
import os
from datetime import datetime, timedelta
import pickle as pkl
from AgensConnector import *
from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)


from .agquery import *
#from agquery import *


class SessionGraph:
    def __init__(self, session):
        self.session = session
        self.path = '/home/airflow/airflow/dags/models/sessiongraph'
        self.knnpath = '/home/airflow/airflow/dags/models/itemknn'
        self.graph_path = 'second_graph'

    def get_top(self,group,topk=100):
        top = group.nlargest(topk, 'weight')
        top['rank'] = range(1, len(top) + 1)
        return top

    def saved_model(self, start_time, end_time):
        
        lap_start = time.time()
        
        self.session.set_graph(self.graph_path) 

        query,create_vertex_query,create_edge_query,loaded_query,aaindex_query =session_query(start_time,end_time)
        session_graph=self.session.query_pandas(query) # Session_graph Load
        item1=list(session_graph.item_idx1.values)
        item2=list(session_graph.item_idx2.values)
        item_set =  list(set(item1).union(item2))
        
        self.session.query_exe(create_vertex_query,(item_set,item_set))
        print('vertex create complete')
        session_loaded = self.session.query_pandas(loaded_query)
        #self.session.query_exe(create_edge_query,(item1,item2))
        session_aaindex= self.session.query_pandas(aaindex_query)
        session_aaindex[['item_idx1', 'item_idx2']] =session_aaindex[['item_idx1', 'item_idx2']].apply(lambda x: x.map(int))
        session_loaded[['item_idx1', 'item_idx2']] = session_loaded[['item_idx1', 'item_idx2']].apply(lambda x: x.map(int))
        print('aaindex complete')
        session_graph.weight = session_graph.weight.apply(lambda x:float(x+100))
        session_loaded.weight = session_loaded.weight.apply(lambda x:float(x+50))
        #df_aaindex.columns = ['item_idx1','item_idx2','weight']
        
        # session_graph  
        result=pd.concat([session_graph,session_aaindex,session_loaded])
        result=result.groupby('item_idx1').apply(self.get_top).reset_index(drop=True)
        result_dict = {value[0]: {'item_idx': value[1], 'aaindex': value[2]} for value in result.values}

        # 파일을 메모리에 매핑
        with open(os.path.join(self.path,'session_result.pkl'), 'rb') as file:
            mapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # 매핑된 파일을 딕셔너리로 로드
        original_dict = pkl.loads(mapped_file)

        # 매핑된 파일로 추천 결과 업데이트
        for key, value in result_dict.items():
            original_dict[key] = value
        
        
        lap_end = time.time()
        sec = (lap_end - lap_start)
        laps_time = timedelta(seconds=sec)
        print(f'finished {start_time} - {end_time} :',laps_time)
        



