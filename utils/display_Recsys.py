#!/usr/bin/env python
#-*- coding: utf-8 -*-

#####################################################
# Program        : display_Recsys.py
# Main function  : get response fastapi & display result images 
# Creator        : Doohee Jung 
# Created date   : 2023.07.07
# Comment        :
#####################################################

import streamlit as st
import pandas as pd
import requests
import json

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]    


class RecDisplay(metaclass=SingletonMeta):
    
    def __init__(self,host,port):
        self.host = host
        self.port = port
        self.api_addr = "http://" + host + ':' + str(port)
        

    def preprocessing(self, df, sizeFilter=False):
        param = dict()
        if len(df) != 0:
            if sizeFilter:
                df = df[df.size_flag == 1]

            df = df.drop_duplicates(["상품명"], keep="first", ignore_index=True)
            df["item_img_path"] = df["item_img_path"].apply(
                lambda x: "http://ccimg.hellomarket.com" + str(x) + "?size=s3"
            )
            param["image_url_list"] = list(df.item_img_path.values)
            param["price_list"] = list(df["가격"].values)
            param["item_cate_list"] = list(df["상품카테고리"].values)
            param["item_name_list"] = list(df["상품명"].values)

            df["상품아이디"] = df["상품아이디"].map(lambda x: "{:.0f}".format(int(x)))
            df.drop(labels="item_img_path", axis=1, inplace=True)

            return param, df
        else:
            param["image_url_list"] = []
            param["price_list"] = []
            param["item_cate_list"] = []
            param["item_name_list"] = []

            return param, df


    def display_image(
        self, image_url_list, price_list, item_cate_list, item_name_list, col_num=5
    ):
        num_images = len(image_url_list)
        num_rows = (num_images - 1) // col_num + 1

        for row in range(num_rows):
            cols = st.columns(col_num)
            start_index = row * col_num
            end_index = min((row + 1) * col_num, num_images)

            for i in range(start_index, end_index):
                with cols[i - start_index]:
                    # 이미지 표시
                    st.image(image_url_list[i], use_column_width=True)

                    # 가격과 상품명 표시
                    st.write(f"##### {price_list[i]:,}원")  # .format(prices[i]))
                    st.write(f"{item_cate_list[i]}")
                    st.write(f"###### {item_name_list[i][:20]}")
    
    def user_rec_view(self,user_id, model, topk=1000, start_date="2023-05-24", end_date="2023-06-01"):
    
        resp_rec = requests.post(
            self.api_addr + "/user_rec_view",
            json={"user_id": int(user_id), "model": model, "topk": topk},
        )
        resp_user = requests.post(
            self.api_addr + "/user_view",
            json={"user_id": int(user_id), "start_date": start_date, "end_date": end_date},
        )
        rec_json = resp_rec.json()
        user_json = resp_user.json()
        rec_view = pd.read_json(rec_json)  
        user_view = pd.read_json(user_json)
        
        rec_param, rec_view = self.preprocessing(rec_view)
        view_param, user_view = self.preprocessing(user_view)

        if len(rec_view) != 0:
            rec_view["상품등록일시"] = pd.to_datetime(rec_view["상품등록일시"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        return rec_param, view_param, rec_view, user_view     


    def item_rec_view(self, itemid, models, topk, sizeFilter):
        resp = requests.post(
            self.api_addr + "/item_rec_view", json={"itemid": int(itemid)}
        )  # ,"model":models})

        itemknn_json, graph_json, view_json = resp.json()

        df_itemknn = pd.read_json(itemknn_json)
        df_graph = pd.read_json(graph_json)
        df_view = pd.read_json(view_json)

        itemknn_param, df_itemknn = self.preprocessing(df_itemknn.head(topk), sizeFilter)
        graph_param, df_graph = self.preprocessing(df_graph.head(topk), sizeFilter)
        view_param, df_view = self.preprocessing(df_view, None)

        return df_itemknn, df_graph, df_view, itemknn_param, graph_param, view_param

