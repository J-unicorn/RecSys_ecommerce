#!/usr/bin/env python
# -*- coding: utf-8 -*-
#####################################################
# Program        : fastapi_Recsys.py
# Main function  : Serving Rec-models result by fastapi
# Creator        : Doohee Jung
# Created date   : 2023.07.07
# Comment        :
#####################################################


import uvicorn
from fastapi import FastAPI, Request

import pandas as pd
import pickle
import sys
import os
import time
import numpy as np
from tempfile import mkdtemp
import json
import yaml
import mmap
import heapq
import argparse
from AgensConnector import *


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class RecFastapi(metaclass=SingletonMeta):
    def __init__(self, host, port, topn_w, lgcn_w, it2v_w, connect_info, model_path):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.topn_w = int(topn_w)
        self.lgcn_w = int(lgcn_w)
        self.it2v_w = int(it2v_w)
        self.connect_info = connect_info
        self.model_path = model_path
        self.norm = 1 / (self.topn_w + self.lgcn_w + self.it2v_w)
        self.agconn = AgensConnector(**self.connect_info)

        # Ensemble User & Item load
        with open(self.model_path + "ensemble/tot_user_dict.pkl", "rb") as f:
            self.total_user_dict = pickle.load(f)
        with open(self.model_path + "ensemble/tot_item_dict.pkl", "rb") as f:
            self.total_item_dict = pickle.load(f)
        self.rev_total_item_dict = {v: k for k, v in self.total_item_dict.items()}

        # Ensemble models memory mapping
        self.lgcn_score = self.read_memmap(
            self.model_path + "ensemble/lightgcn_aligned_score"
        )
        self.topn_score = self.read_memmap(
            self.model_path + "ensemble/topn_aligned_score"
        )
        self.it2v_score = self.read_memmap(
            self.model_path + "ensemble/item2vec_aligned_score"
        )

        # session_graph Pickle load
        with open(self.model_path + "sessiongraph/session_result.pkl", "rb") as file:
            mapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self.session_result = pickle.loads(mapped_file)

        # Itemknn Pickle load
        self.itemknn_index = np.load(self.model_path + "itemknn/itemknn_index.npy")
        with open(self.model_path + "itemknn/rev_itemknn_dict.pkl", "rb") as f:
            self.rev_itemknn_dict = pickle.load(f)
        self.itemknn_dict = {v: k for k, v in self.rev_itemknn_dict.items()}

    def run_server(self):
        # @app.get("/")
        # def root():
        #     return {"message": "Hello, World!"}

        # @app.get("/info")
        # def get_info():
        #     return {"info": "This is an RecSys API server."}

        uvicorn.run(self.app, host=self.host, port=self.port)

    def read_memmap(self, mem_file_name):
        """디스크에 저장된 numpy.memmap객체를 읽는다"""
        # r+ mode: Open existing file for reading and writing
        with open(mem_file_name + ".conf", "r") as file:
            memmap_configs = json.load(file)
            return np.memmap(
                mem_file_name,
                mode="r+",
                shape=tuple(memmap_configs["shape"]),
                dtype=memmap_configs["dtype"],
            )

    def item_rec_view(self, itemid):
        view_query = f"""
                    select T.* , 
                        tbl_item_img.item_img_path,
                        CASE 
                            WHEN tbl_item_clothes_size.clothes_size_idx IS NOT NULL 
                            THEN 'tbl_item_clothes_size.clothes_size_idx'
                            WHEN tbl_item_shoes_size.shoes_size_idx IS NOT NULL 
                            THEN 'tbl_item_shoes_size.shoes_size_idx'
                            ELSE '00' END AS SIZE_FLAG,
                        CASE 
                            WHEN tbl_item_clothes_size.clothes_size_idx IS NOT NULL 
                            THEN tbl_item_clothes_size.clothes_size_idx
                            WHEN tbl_item_shoes_size.shoes_size_idx IS NOT NULL 
                            THEN tbl_item_shoes_size.shoes_size_idx 
                        ELSE '00' END AS SIZE_IDX
                    from tbl_streamlit_item_detail_view T
                    INNER JOIN tbl_item_img 
                    ON 상품아이디 = tbl_item_img.item_idx
                    LEFT OUTER JOIN tbl_item_shoes_size
                    ON t.상품아이디 = tbl_item_shoes_size.item_idx
                    LEFT OUTER JOIN tbl_item_clothes_size
                    ON t.상품아이디 = tbl_item_clothes_size.item_idx
                    where 상품아이디 = {itemid}
                    """
        df_view = self.agconn.query_pandas(view_query)
        # df_view['상품등록일시']= df_view['상품등록일시'].dt.strftime('%Y-%m-%d %H:%M:%S')

        size_flag, size_idx = df_view.size_flag[0], df_view.size_idx[0]

        df_view.drop(labels=["size_flag", "size_idx"], axis=1, inplace=True)

        # if model == 'Item2Vec':

        pred_idx = self.itemknn_dict[itemid]
        pred_idx = self.itemknn_index[pred_idx][:100]
        pred_idx = [self.rev_itemknn_dict[i] for i in pred_idx]
        base_query = f"""
                    SELECT T.*, 
                        tbl_item_img.item_img_path,
                        CASE 
                            WHEN tbl_item_clothes_size IS NOT NULL 
                            AND tbl_item_clothes_size.clothes_size_idx={size_idx} 
                            THEN '01'
                            WHEN tbl_item_shoes_size IS NOT NULL 
                            AND tbl_item_shoes_size.shoes_size_idx={size_idx} 
                            THEN '01'
                            WHEN tbl_item_shoes_size.shoes_size_idx IS NULL 
                            AND tbl_item_clothes_size.clothes_size_idx IS NULL
                            THEN '01'
                            ELSE '02' 
                            END AS SIZE_FLAG
                    FROM tbl_streamlit_item_detail_view T 
                    INNER JOIN tbl_item_img 
                    ON 상품아이디 = tbl_item_img.item_idx
                    INNER JOIN (
                                SELECT unnest(
                                            ARRAY {pred_idx}
                                    ) AS item_idx
                               ) AS ids ON T.상품아이디 = ids.item_idx
                    LEFT OUTER JOIN tbl_item_shoes_size
                    ON 상품아이디 = tbl_item_shoes_size.item_idx
                    LEFT OUTER JOIN tbl_item_clothes_size
                    ON 상품아이디 = tbl_item_clothes_size.item_idx
                    ORDER BY CASE tbl_item_img.item_idx
                    """
        query = base_query
        for i, k in enumerate(pred_idx):
            query = query + f"""WHEN {k} THEN {i+1} """
        item2vec_query = query + "ELSE 9999 END"
        # elif model =='Session_graph':
        df_item2vec = self.agconn.query_pandas(item2vec_query)

        session_predict = self.session_result[itemid]
        session_predict_100 = heapq.nlargest(
            100, session_predict, key=session_predict.get
        )

        base_query = f"""
                        SELECT T.*,
                            tbl_item_img.item_img_path,
                            CASE 
                                WHEN tbl_item_clothes_size IS NOT NULL 
                                AND tbl_item_clothes_size.clothes_size_idx={size_idx} 
                                THEN '01'
                                WHEN tbl_item_shoes_size IS NOT NULL 
                                AND tbl_item_shoes_size.shoes_size_idx={size_idx} 
                                THEN '01'
                                WHEN tbl_item_shoes_size.shoes_size_idx IS NULL 
                                AND tbl_item_clothes_size.clothes_size_idx IS NULL
                                THEN '01'
                                ELSE '02' 
                                END AS SIZE_FLAG
                        FROM tbl_streamlit_item_detail_view T 
                        INNER JOIN tbl_item_img 
                        ON 상품아이디 = tbl_item_img.item_idx
                        INNER JOIN (
                                    SELECT unnest(
                                        ARRAY {session_predict_100}
                                        ) AS item_idx
                                    ) AS ids 
                        ON T.상품아이디 = ids.item_idx
                        LEFT OUTER JOIN tbl_item_shoes_size
                        ON 상품아이디 = tbl_item_shoes_size.item_idx
                        LEFT OUTER JOIN tbl_item_clothes_size
                        ON 상품아이디 = tbl_item_clothes_size.item_idx
                        ORDER BY CASE tbl_item_img.item_idx
                        """
        if len(session_predict_100) > 0:
            query = base_query
            for i, k in enumerate(session_predict_100):
                query = query + f"""WHEN {k} THEN {i+1} """
            sessiongraph_query = query + "ELSE 9999 END"
            df_graph = self.agconn.query_pandas(sessiongraph_query)
        else:
            df_graph = df_item2vec.head(0)

        if len(df_item2vec) != 0:
            df_item2vec["상품등록일시"] = df_item2vec["상품등록일시"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        if len(df_graph) != 0:
            df_graph["상품등록일시"] = df_graph["상품등록일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
        if len(df_view) != 0:
            df_view["상품등록일시"] = df_view["상품등록일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
            
        return df_item2vec.to_json(), df_graph.to_json(), df_view.to_json()

    def item_views(self, itemid):
        query = f"""
                select T.* ,item_img.item_img_path 
                from tbl_streamlit_item_detail_view T
                INNER JOIN tbl_item_img 
                ON 상품아이디 = tbl_item_img.item_idx
                where 상품아이디 = {itemid} 
                """
        df_view = self.agconn.query_pandas(query)
        df_view["상품등록일시"] = df_view["상품등록일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df_view.to_json()

    def user_view(self, user_id, start_date, end_date):
        view_query = f"""
                
                select T1.*, T2.ITEM_IMG_PATH
                from tbl_streamlit_user_view T1
                INNER join tbl_ITEM_IMG T2
                on T2.ITEM_IDX = T1.상품아이디
                AND 유저아이디 = {user_id} 
                and 조회일시 between '{start_date}'
                                and '{end_date}'
                            order by 조회일시 desc limit 50
                    """

        df_view = self.agconn.query_pandas(view_query)

        if len(df_view) != 0:
            df_view["상품등록일시"] = df_view["상품등록일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_view["조회일시"] = df_view["조회일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df_view.to_json()

    def user_rec_view(self, user_id, model, topk):
        global user_score

        user_idx = self.total_user_dict[int(user_id)]
        if model == "LightGCN":
            user_score = self.lgcn_score[user_idx]
        elif model == "TopN":
            user_score = self.topn_score[user_idx]
        elif model == "Item2Vec":
            user_score = self.it2v_score[user_idx]
        elif model == "Ensemble":
            user_score = (
                self.lgcn_score[user_idx]
                + self.topn_score[user_idx]
                + self.it2v_score[user_idx]
            ) * self.norm
        # user_score[view_ids] = -np.inf
        indices = user_score.argsort()[::-1][:topk]
        pred_idx = [self.rev_total_item_dict[i] for i in indices]

        query = ""
        image_query = f"""
                        SELECT T.*,
                          tbl_item_img.item_img_path
                        FROM tbl_streamlit_item_view T 
                        INNER JOIN tbl_item_img 
                        ON 상품아이디 = tbl_item_img.item_idx
                        INNER JOIN (
                                SELECT unnest(
                                        ARRAY {pred_idx}
                                        ) AS item_idx
                        ) AS ids 
                        ON tbl_item_img.item_idx = ids.item_idx
                        WHERE NOT EXISTS (
                                    SELECT * 
                                    FROM tbl_streamlit_user_view 
                                    WHERE 유저아이디 = {user_id}
                                    AND tbl_streamlit_user_view.상품아이디 = T.상품아이디 
                                        )
                        ORDER BY CASE tbl_item_img.item_idx
                        """

        for i, k in enumerate(pred_idx):
            query = query + f"""WHEN {k} THEN {i+1} """
        image_query = image_query + query + "ELSE 9999 END"
        if sum(user_score) == 0:
            image_query = image_query + " limit 0"

        df_rec = self.agconn.query_pandas(image_query)

        if len(df_rec) != 0:
            df_rec["상품등록일시"] = df_rec["상품등록일시"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return df_rec.to_json()

    def test_view(self, user_id, start_date, end_date):
        query = f"""
                    select * 
                    from tbl_streamlit_user_view
                    where 유저아이디 = {user_id} 
                    and 조회일시 between '{start_date}' 
                                     and '{end_date}'
                    """
        df_test = self.agconn.query_pandas(query)
        df_test["상품등록일시"] = df_test["상품등록일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_test["조회일시"] = df_test["조회일시"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df_test.to_json()


if __name__ == "__main__":
    # YAML 파일 로드
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # 설정 가져오기
    connect_info = config["connect_info"]
    model_weights = config["model_weights"]
    fastapi_info = config["fastapi_info"]
    path_info = config["path_info"]

    # 모델 가중치 설정 가져오기
    topn_w = int(model_weights["topn"])
    lgcn_w = int(model_weights["lgcn"])
    it2v_w = int(model_weights["it2v"])

    # FastAPI 정보 가져오기
    host = fastapi_info["host"]
    port = fastapi_info["port"]

    # 모델 정보 가져오기
    base_path = path_info["base_path"]
    model_path = base_path + path_info["model_path"]
    page_path = base_path + path_info["page_path"]

    # RecFastapi 인스턴스 생성

    rec_fastapi = RecFastapi(
        host=host,
        port=port,
        topn_w=topn_w,
        lgcn_w=lgcn_w,
        it2v_w=it2v_w,
        connect_info=connect_info,
        model_path=model_path,
    )
    print(host,port,connect_info,model_path )
    @rec_fastapi.app.post("/item_rec_view")
    async def item_rec_view_api(request: Request):
        data = await request.json()
        itemid = data["itemid"]
        # model = data['model']
        result = rec_fastapi.item_rec_view(itemid)
        return result

    @rec_fastapi.app.post("/user_view")
    async def user_view_api(request: Request):
        data = await request.json()
        user_id = data["user_id"]
        start_date = data["start_date"]
        end_date = data["end_date"]
        result = rec_fastapi.user_view(user_id, start_date, end_date)
        return result

    @rec_fastapi.app.post("/user_rec_view")
    async def user_rec_view_api(request: Request):
        data = await request.json()
        user_id = data["user_id"]
        model = data["model"]
        topk = data["topk"]
        result = rec_fastapi.user_rec_view(user_id, model, topk)
        return result

    @rec_fastapi.app.post("/test_view")
    async def test_view_api(request: Request):
        data = await request.json()
        user_id = data["user_id"]
        start_date = data["start_date"]
        end_date = data["end_date"]
        result = rec_fastapi.test_view(user_id, start_date, end_date)
        return result
    
    rec_fastapi.run_server()
