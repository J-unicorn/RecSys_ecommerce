import os
import streamlit as st
import pickle
import pandas as pd
import random
import yaml
import sys
sys.path.append('/home/agens/conda_user/hello/pages')
from utils.display_Recsys import *

import streamlit_toggle as tog


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



class StreamlitItemRecApp:
    def __init__(self,host,port,model_path):
        self.session_state = SessionState(item_id=176919591,model=None,topk=None,sizeFilter=False) 
        self.display_Recsys = RecDisplay(host,port)
        self.model_path = model_path
        with open(self.model_path + "sessiongraph/item_list.pkl", "rb") as f:
            self.item_list = pickle.load(f)

    def run(self):

        # 하기 내용은 표시 텍스트
        st.set_page_config(
            page_title="SecondWear 상품상세추천 테스트", layout="wide", initial_sidebar_state="expanded"
        )

        st.subheader("""상품상세 추천 결과입니다 \n 아이템을 조회하여 주시고 추천모델을 선택해주세요""")



        col1, col2 = st.columns([0.85, 0.15])

        with col1:
            random_button = st.button(label="무작위 아이템 ID 선택")

        with col2:
            self.session_state.sizeFilter = tog.st_toggle_switch(
                label="SizeFilter",
                # key="Key1",
                default_value=False,
                label_after=False,
                inactive_color="#D3D3D3",
                active_color="#11567f",
                track_color="#29B5E8",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            if random_button:
                self.session_state.item_id = random.sample(self.item_list, 1)[0]
            st.text_input("아이템 ID를 입력하세요:", value=self.session_state.item_id)

        with col2:
            self.session_state.model = st.selectbox("추천 모델을 선택하세요:", ["Item2Vec", "Session_graph"])


        with col3:
            self.session_state.topk = st.number_input(
                "추천받을 아이템 수를 입력하세요:", format="%g", step=1, min_value=1, max_value=100, value=100
            )

            if self.session_state.item_id and self.session_state.model and self.session_state.topk:
                (
                    df_itemknn,
                    df_graph,
                    df_view,
                    itemknn_param,
                    graph_param,
                    view_param,
                ) = self.display_Recsys.item_rec_view(
                    self.session_state.item_id,
                    self.session_state.model,
                    self.session_state.topk,
                    self.session_state.sizeFilter,
                )


        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            if self.session_state.item_id:
                self.display_Recsys.display_image(**view_param, col_num=1)

        with col2:
            if self.session_state.item_id:
                st.dataframe(df_view, use_container_width=True)


        col1, col2 = st.columns(2)
        with col1:
            with st.expander("#### 유사 상품 추천 이미지", expanded=True):
                if self.session_state.item_id:
                    self.display_Recsys.display_image(**itemknn_param, col_num=5)

        with col2:
            with st.expander("#### 함께본 상품 추천 이미지", expanded=True):
                if self.session_state.item_id:
                    if len(graph_param["image_url_list"]) == 0:
                        st.write("함께본 조회가 존재하지않습니다")
                    else:
                        # 한 row에 5개의 이미지를 표시하기 위해, 5개의 column 생성
                        self.display_Recsys.display_image(**graph_param, col_num=5)
                    

        with st.expander("추천 결과 보기"):
            if self.session_state.item_id and self.session_state.model:
                rec_view = df_itemknn if self.session_state.model == "Item2Vec" else df_graph
                st.dataframe(rec_view, use_container_width=True)



if __name__ == "__main__":
    # YAML 파일 로드
    with open("/home/agens/conda_user/hello/config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    # 설정 가져오기
    model_weights = config["model_weights"]
    fastapi_info = config["fastapi_info"]
    path_info = config["path_info"]
    
    host = fastapi_info["host"]
    port = fastapi_info["port"]
    base_path = path_info['base_path']
    model_path = base_path + path_info["model_path"]
    page_path = base_path + path_info["page_path"]

    sys.path.append(page_path)

    app = StreamlitItemRecApp(host=host,port=port,model_path=model_path)
    app.run()