import os
import streamlit as st
import subprocess
import sys
import time
import webbrowser
import yaml

log_file_path = "fastapi.log"

# 하기 내용은 표시 텍스트
st.set_page_config(
    page_title="SecondWear 추천테스트", layout="wide", initial_sidebar_state="expanded"
)

st.title("SecondWear 추천 테스트 페이지")
st.text("")
st.text("")
st.subheader("""홈피드와 상세 추천 결과를 확인할 수 있는 페이지입니다.""")

st.text("")
st.text("")


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
model_weights = config["model_weights"]

session_state = SessionState(**model_weights)
topn_w = int(model_weights["topn"])
lgcn_w = int(model_weights["lgcn"])
it2v_w = int(model_weights["it2v"])


def session_update(session_state):
    model_weights.update(vars(session_state))
    # config 업데이트
    config["model_weights"] = model_weights
    with open("config.yml", "w") as file:
        yaml.safe_dump(config, file)


# def spinner():
#     subprocess.run(
#         path
#         + f"start_fastapi.sh {session_state.topn} {session_state.lgcn} {session_state.item2vec}",
#         shell=True,
#     )  # 방법1

#     with open(log_file_path, "r") as file:
#         success = False
#         while not success:
#             line = file.readline()
#             with st.spinner("재기동하는데 시간이 걸릴 수 있습니다."):
#                 time.sleep(10)

#             if "Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)" in line:
#                 success = True

#         st.success("성공하였습니다.")


c1, c2 = st.columns(2)
with c1:
    with st.expander("#### 추천 모델 관리"):
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            api_button = st.button(label="추천 모델 API 재기동")

        with col2:
            st.markdown("###### 추천모델 API를 재시작합니다.")
            if api_button:
                pass
                #spinner()
        st.write("")
        st.write("")

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            weight_button = st.button(label="홈피드 가중치 재설정")
        with col2:
            st.markdown("###### 홈피드 Ensemble 모델 가중치를 재설정합니다.")
            if weight_button:
                pass
                #spinner()
            st.markdown("###### 현재 모델 가중치 현황")

        col0, col1, col2, col3 = st.columns([0.3, 0.2, 0.2, 0.2])

        with col1:
            session_state.topn = st.number_input(
                "TopN", format="%g", step=1, min_value=1, value=session_state.topn
            )
            session_update(session_state)
        with col2:
            session_state.item2vec = st.number_input(
                "Item2vec",
                format="%g",
                step=1,
                min_value=1,
                value=session_state.item2vec,
            )
            session_update(session_state)
        with col3:
            session_state.lgcn = st.number_input(
                "GraphNN", format="%g", step=1, min_value=1, value=session_state.lgcn
            )
            session_update(session_state)

with c2:
    with st.expander("#### GraphDB 및 Airflow 관리"):
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            db_start_button = st.button(label="GraphDB 재기동")

        with col2:
            st.markdown("###### (개발중)GraphDB를 재시작합니다.")
            if db_start_button:
                pass

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            airflow_start_button = st.button(label="Airflow 재기동")

        with col2:
            st.markdown("###### (개발중)Airflow를 재시작합니다.")
            if airflow_start_button:
                pass

        # col1, col2 = st.columns([0.3,0.7])
        # with col1:
        #     airflow_link_button = st.button(label='Airflow Web')

        # with col2:
        #     st.markdown('###### Airflow를 Web Page를 엽니다.')
        #     if airflow_link_button:
        #         webbrowser.open_new_tab('https://127.0.0.1:8080/home')0


# with col2:
#     with st.expander('#### 함께본 상품 추천 이미지',expanded=True):
