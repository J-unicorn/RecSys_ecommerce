#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mmap
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle as pkl
from AgensConnector import *
from multiprocessing import Pool, Manager
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
import re
import multiprocessing
cores = multiprocessing.cpu_count()
import os
# 병렬 처리에 사용할 스레드 수 설정
os.environ['OMP_NUM_THREADS'] = str(cores-2)

from .agquery import *
#from agquery import *


class ItemKNN:
    def __init__(self, session):
        self.session = session
        self.path = '/home/airflow/airflow/dags/models/itemknn'

    
    def text_preprocessor(self,s):
        
        ## (3) 특수문자 제거
        pattern = r'[^a-zA-Z가-힣]'
        s = re.sub(pattern=pattern, repl=' ', string=s)
        s = s.lower()
            
        # (5) 공백 기준으로 분할하기
        s_split = s.split()
    
        s_list = []
        for word in s_split:
            if len(word) !=1:
                s_list.append(word)
                
        return s_list


    def train_mode(self,df,mode=['d2v','ft']):
        if mode == 'd2v':
            import gensim
            from gensim.models.doc2vec import FAST_VERSION ,Doc2Vec, TaggedDocument
            print('doc2vec model train!')
            taggedDocs = [TaggedDocument(words= set ( i ), tags=[f"{i}"])  for i in df]
            model = Doc2Vec(taggedDocs, vector_size=300, window=8, min_count=2, workers=cores)
            return df.apply(lambda x : model.infer_vector(x))
        if mode == 'ft':
            import fasttext,os
            model_path = 'models/itemknn_model.bin'
            if os.path.isfile(model_path):
                model = fasttext.load_model(model_path)
            else:   
                print('fasttext model train!')
                df.to_csv('train.txt', sep = '\t', index = False)
                model = fasttext.train_unsupervised('train.txt', lr=0.1, epoch=200, dim=300,ws=20,minn=3,thread=cores)
                model.save_model(model_path)
            return df.apply(lambda x : model.get_sentence_vector(','.join(x)))
        else :
            print('select train model !')
            quit()


    def saved_model(self, start_time, end_time):
        
        global get_recommendations
    
        
        item_meta=self.session.query_pandas(itemknn_query(start_time, end_time))
      
        item_dict = {k:i for i,k in enumerate(item_meta.item_idx.unique())}
        rev_item_dict = {i:k for i,k in enumerate(item_meta.item_idx.unique())}
        num_item = len(item_meta)

        #target_item 추출 ( 5명에게 조회 및 
        target_indice = np.where(item_meta.flag.values == 1)[0]
        print('# of item :' ,num_item)
        print('# of target_item : ',len(target_indice))
    
        item_meta.item_idx = item_meta.item_idx.apply(lambda x : item_dict[x])
        item_meta['item_title'] = item_meta['item_title'].apply(lambda x : self.text_preprocessor(x))
        item_meta['item_tag']=[set ( v+[x]+ y + z ) for i,v,x,y,z in zip (item_meta['item_idx'],item_meta['item_brand'],item_meta['item_category'],item_meta['item_tag'],item_meta['item_title'])]
        item_meta['item_tag']=item_meta['item_tag'].apply(lambda item: [x.replace('\n', '') for x in item])
    
        item_meta['vector']  = self.train_mode(item_meta['item_tag'],'ft')
    
        vec = np.array([i for i in item_meta.vector])
        item_norms = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / item_norms 
    
        print('finish generate vector')    
        reducek = 30
        pca = PCA(n_components=reducek)
        reduced_vectors = pca.fit_transform(vec)
    
        print('finish reduce vector')
        tree_count = 200
        annoy_index = AnnoyIndex(reducek, 'angular')
        for i in target_indice:
            annoy_index.add_item(i, reduced_vectors[i])
        annoy_index.build(tree_count)
    
        print('finish annoy build')

        def get_recommendations(target_item):
        # 대상 상품에 대한 추천 목록 생성
            target_vector = reduced_vectors[target_item]
            topk = 200
            #knn_indices,knn_distances = annoy_index.get_nns_by_vector(target_vector, topk +1 ,include_distances=True)
            
            knn_indices = annoy_index.get_nns_by_vector(target_vector, topk +1)# 자기 자신을 제외한 상위 k개 추천
            
            #knn_distances, knn_indices = annoy_index.get_nns_by_item(item_idx, k+1, search_k=-1, )
            #annoy_recommendations = [ rev_item_dict[i] for i in annoy_recommendations]
            # 추천 목록 합치기
            #knn_distances = knn_distances[1:]
            knn_indices = knn_indices[1:]
            
            # knn_distances를 0.5에서 1 사이의 값으로 변환
            #min_distance = min(knn_distances)
            #max_distance = max(knn_distances)
            #range_distance = max_distance - min_distance
            scores = [(1 - i /2/len(knn_indices)) for i in range(len(knn_indices))]  
            #knn_distances = [ round((1 - distance/max_distance) * 0.5 + 0.5,4) for distance in knn_distances]
            return scores, knn_indices 

        manager = Manager()
        results = []
        pool = Pool(cores-2)
        for target_item in range(len(item_meta)):
            result = pool.apply_async(get_recommendations, args=(target_item,))
            results.append(result)
        
        pool.close()
        pool.join()
        
        final_results = [result.get() for result in results]
    
        print('finish join')    

        def generate_itemknn(final_results):
            for item in final_results:
                yield item
        itemknn_generator = generate_itemknn(final_results)
        
        itemknn_score, itemknn_index = zip(*itemknn_generator)
    
        
        
        del final_results
        del results
        itemknn_score = np.array(itemknn_score, dtype=np.float16)
        itemknn_index = np.array(itemknn_index, dtype=np.int32)
            
        print('finish ndarray')    
        
        np.save('models/itemknn/itemknn_score',itemknn_score)
        np.save('models/itemknn/itemknn_index',itemknn_index)
            
        with open(file='models/itemknn/rev_itemknn_dict.pkl', mode='wb') as f:
             pkl.dump(rev_item_dict, f)

