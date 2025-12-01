# Protease Simulator 3

단백질 분해 효소(Protease) 반응 시뮬레이션 및 데이터 분석 도구입니다.

## 주요 기능

*   **Data Load 모드**: 실험 데이터(CSV/XLSX) 로드 및 전처리
    *   Substrate 농도 변화 및 Enzyme 농도 변화 실험 지원
    *   Michaelis-Menten 모델 피팅 및 초기 속도(v0) 계산
    *   데이터 보간(Interpolation) 및 정규화(Normalization)
*   **General Analysis 모드**: 분석된 데이터 시각화 및 비교 분석
*   **Data Interpolation**: Prism 스타일의 고품질 보간 곡선 생성

## 설치 방법

1.  Python 3.8 이상 설치
2.  필요한 패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```

## 실행 방법

```bash
python -m streamlit run app.py
```

## 데이터 형식

*   **prep_raw.csv/xlsx**:
    *   Column 1: Time (min)
    *   Column 2, 5, 8...: Value (농도별)
    *   Column 3, 6, 9...: SD (옵션)
    *   Column 4, 7, 10...: N (옵션)
    *   헤더에 각 농도 정보 포함 필요

