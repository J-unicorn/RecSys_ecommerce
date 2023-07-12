import os
import streamlit as st
import pickle
import random
import yaml
import numpy as np
import sys
from utils.display_Recsys import *

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



class StreamlitUserRecApp:
    def __init__(self,host,port,model_path):
        self.session_state = SessionState(user_id=5017208, model="Ensemble", topk=100)
        self.display_Recsys = RecDisplay(host,port)
        self.model_path = model_path
        with open(self.model_path + "/tot_user_dict.pkl", "rb") as f:
            self.total_user_dict = list(pickle.load(f).keys())

    def run(self):
        st.set_page_config(
            page_title="개인화 추천 테스트 페이지", layout="wide", initial_sidebar_state="expanded"
        )


        col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
        with col1:
            st.subheader("""개인화 추천 결과 조회하기""")
            st.write("회원 ID와 추천 모델, 추천 아이템 수를 입력하세요.")

            random_button = st.button(label="### Random 회원 ID 선택")

        col1, col2, col3 = st.columns(3)

        with col1:
            if random_button:
                self.session_state.user_id = random.sample(self.total_user_dict, 1)[0]
            self.session_state.user_id = st.text_input("회원 ID를 입력하세요: ", value=self.session_state.user_id)
            
        with col2:
            self.session_state.model = st.selectbox(
                "추천 모델:", ["Ensemble", "LightGCN", "TopN", "Item2Vec"]
            )
        
        with col3:
            self.session_state.topk = st.number_input(
                "추천 아이템 수:", format="%g", step=1, min_value=1, value=100
            )
            if self.session_state.user_id and self.session_state.model and self.session_state.topk:
                rec_param, view_param, rec_view, user_view = self.display_Recsys.user_rec_view(
                    self.session_state.user_id, self.session_state.model, self.session_state.topk
                )

        # st.markdown("___")
        st.write("")


        col1, col2 = st.columns(2)
        # expander 생성
        with col1:
            with st.expander(f"#### 조회상품 이미지", expanded=True):
                if self.session_state.user_id:
                    if len(view_param["image_url_list"]) == 0:
                        st.write("조회상품의 이미지 파일을 찾지 못하였습니다.")
                    else:
                        # 한 row에 5개의 이미지를 표시하기 위해, 5개의 column 생성
                        self.display_Recsys.display_image(**view_param, col_num=5)


        with col2:
            with st.expander(f"#### 추천상품 이미지", expanded=True):
                if self.session_state.user_id:
                    # 한 row에 5개의 이미지를 표시하기 위해, 5개의 column 생성
                    if len(rec_param["image_url_list"]) == 0:
                        st.write(
                            f""" 사용자 {self.session_state.user_id}는 모델 {self.session_state.model}에서 추천에 포함되지 않았습니다."""
                        )
                    else:
                        # 한 row에 3개의 이미지를 표시하기 위해, 3개의 column 생성
                        self.display_Recsys.display_image(**rec_param, col_num=5)


        with st.expander(f"**추천 결과**", expanded=True):
            if self.session_state.user_id:
                st.dataframe(rec_view, height=200, use_container_width=True)

        with st.expander(f"**학습 기간 회원 조회 기록**"):
            if self.session_state.user_id:
 
                st.dataframe(user_view, height=200, use_container_width=True)
        

if __name__ == "__main__":
    # YAML 파일 로드
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    # 설정 가져오기
    model_weights = config["model_weights"]
    fastapi_info = config["fastapi_info"]
    path_info = config["path_info"]
    
    host = fastapi_info["host"]
    port = fastapi_info["port"]
    base_path = path_info['base_path']
    model_path = path_info["model_path"]
    page_path = path_info["page_path"]

    sys.path.append(page_path)
   
    app = StreamlitUserRecApp(host=host,port=port,model_path=model_path)
    app.run()