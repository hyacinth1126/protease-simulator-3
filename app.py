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
        page_title="Hydrogel FRET Advanced Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬  FRET Protease Simulation")
    st.markdown("---")
    
    # Mode selection
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        ["Data Load Mode", "Model Simulation Mode"],
        help="Data Load Mode: Upload CSV file or extract data from image | Model Simulation Mode: Standard FRET analysis"
    )
    # Always render footer at bottom
    render_footer()
    
    # Data Load Mode
    if analysis_mode == "Data Load Mode":
        data_load_mode(st)
        return
    
    # Model Simulation Mode
    general_analysis_mode(st)


if __name__ == "__main__":
    main()
