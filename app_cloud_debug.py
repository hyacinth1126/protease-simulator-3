#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Community Cloud 디버깅용 최소 앱.
Cloud에서 "Main file"을 이 파일로 잠시 바꿔 배포해 보세요.
- 이 앱이 뜨면: 문제는 app.py 또는 의존성 로드 쪽입니다.
- 이 앱도 안 뜨면: 문제는 Cloud 환경/Streamlit 설치 쪽입니다.
"""
import sys
from datetime import datetime

def _log(msg: str) -> None:
    try:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sys.stderr.write(f"[{ts}] [app_cloud_debug] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass

_log("loading: importing streamlit")
import streamlit as st
_log("loading: streamlit imported")

st.set_page_config(page_title="Cloud Debug", page_icon="🔧", layout="centered")
_log("set_page_config done")
st.title("🔧 Cloud OK")
st.write("이 화면이 보이면 Streamlit Cloud 환경은 정상입니다. 메인 앱은 `app.py`로 되돌려 배포하세요.")
_log("render done")
