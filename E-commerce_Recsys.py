import os
import streamlit as st
import subprocess
import sys
import time
from PIL import Image

# 이미지 파일 열기
arc_image = Image.open('data/architecture.png')
model_image = Image.open('data/rec_list.png')



# 하기 내용은 표시 텍스트
st.set_page_config(
    page_title="추천테스트페이지", layout="wide", initial_sidebar_state="expanded"
)

st.title("추천 결과 테스트 페이지")
st.text("")
st.text("")
st.subheader("""E-commerce Fashion Domain에서 개인화 추천과 연관상품 추천 결과를 확인할 수 있는 페이지입니다.""")

st.text("")
st.text("")


with st.expander("#### 추천 시스템 아키텍처 확인하기",expanded=True):
    # 이미지 출력
    st.image(arc_image, caption='추천 시스템 아키텍쳐')
    
with st.expander("#### 추천 모델 비교하기 ",expanded=True):
    # 이미지 출력
    st.image(model_image, caption='추천 모델 비교')

    
    
