#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# .\.venv\Scripts\python.exe -m streamlit run app.py .\.venv\Scripts\python.exe -m streamlit run app.py
# author: hyacinth1126
"""
Hydrogel FRET Advanced Kinetic Analysis - Streamlit Application
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# UI 모듈 import
from app_ui.data_load_mode import data_load_mode
from app_ui.general_analysis_mode import general_analysis_mode
from app_ui.footer import render_footer

# Configure plotting
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="하이드로겔 FRET 고급 분석",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬  FRET Protease Simulation")
    st.markdown("---")
    
    # 모드 선택
    analysis_mode = st.sidebar.radio(
        "분석 모드 선택",
        ["Data Load 모드", "Model Simulation 모드"],
        help="Data Load 모드: CSV 파일 업로드 또는 이미지에서 데이터 추출 | Model Simulation 모드: 표준 FRET 분석"
    )
    # 항상 하단에 푸터 렌더링
    render_footer()
    
    # Data Load 모드
    if analysis_mode == "Data Load 모드":
        data_load_mode(st)
        return
    
    # Model Simulation 모드
    general_analysis_mode(st)


if __name__ == "__main__":
    main()
