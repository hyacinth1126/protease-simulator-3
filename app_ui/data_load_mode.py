import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from mode_prep_raw_data.prep import (
    read_raw_data,
    fit_time_course,
    fit_calibration_curve,
    michaelis_menten_calibration
)
from data_interpolation_mode.interpolate_prism import (
    exponential_association,
    create_prism_interpolation_range
)


def detect_lines_and_points(image_array):
    """
    이미지에서 선과 점을 감지하는 함수
    """
    if not CV2_AVAILABLE:
        return None, None
    
    try:
        # 그레이스케일 변환
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 선 감지 (HoughLinesP)
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        # 점 감지 (contour 기반)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # 점 크기 범위
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        return lines, points
    except Exception as e:
        st.warning(f"자동 감지 오류: {e}")
        return None, None


def extract_line_data_from_image(image_file, lines):
    """
    이미지에서 선 데이터를 추출하고 exponential association 모델로 fitting
    """
    try:
        image = Image.open(image_file)
        img_array = np.array(image)
        
        if lines is None or len(lines) == 0:
            return None
        
        # 선에서 데이터 포인트 추출 (간단한 예시)
        # 실제로는 좌표 변환 및 축 스케일 추출이 필요
        st.info("💡 선 데이터 추출: Exponential Association 모델로 fitting합니다.")
        
        # 여기서는 수동 입력으로 대체
        return None
        
    except Exception as e:
        st.error(f"선 데이터 추출 오류: {e}")
        return None


def extract_legend_text(image_array):
    """
    이미지에서 범례 텍스트를 추출하는 함수 (OCR 사용)
    """
    legend_texts = []
    
    if TESSERACT_AVAILABLE:
        try:
            # 범례 영역은 보통 이미지의 오른쪽 상단 또는 하단에 위치
            # 전체 이미지에서 텍스트 추출 시도
            data = pytesseract.image_to_data(image_array, output_type=Output.DICT, lang='eng')
            
            # 텍스트가 있는 영역 찾기
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if text and conf > 30:  # 신뢰도 30 이상
                    # 농도 관련 텍스트 패턴 찾기 (예: "5 ug/mL", "0.5", "10μM" 등)
                    import re
                    # 숫자와 단위가 포함된 텍스트 찾기
                    if re.search(r'\d+\.?\d*\s*(ug/mL|μg/mL|μM|uM|mg/mL|mM|%)', text, re.IGNORECASE) or \
                       re.search(r'^\d+\.?\d*$', text):
                        legend_texts.append(text)
        except Exception as e:
            st.warning(f"Tesseract OCR 오류: {e}")
    
    if EASYOCR_AVAILABLE and len(legend_texts) == 0:
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(image_array)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 신뢰도 0.5 이상
                    import re
                    # 농도 관련 텍스트 패턴 찾기
                    if re.search(r'\d+\.?\d*\s*(ug/mL|μg/mL|μM|uM|mg/mL|mM|%)', text, re.IGNORECASE) or \
                       re.search(r'^\d+\.?\d*$', text):
                        legend_texts.append(text)
        except Exception as e:
            st.warning(f"EasyOCR 오류: {e}")
    
    return legend_texts


def convert_image_coords_to_data(x_img, y_img, img_width, img_height, 
                                  x_min, x_max, y_min, y_max,
                                  plot_x_min, plot_x_max, plot_y_min, plot_y_max):
    """
    이미지 좌표를 실제 데이터 좌표로 변환
    
    Args:
        x_img, y_img: 이미지 상의 픽셀 좌표
        img_width, img_height: 이미지 전체 크기
        x_min, x_max, y_min, y_max: 그래프 축의 실제 데이터 범위
        plot_x_min, plot_x_max, plot_y_min, plot_y_max: 그래프 영역의 픽셀 좌표
    """
    # Y축은 이미지 좌표계에서 위가 0이므로 반전 필요
    y_img_flipped = img_height - y_img
    
    # 그래프 영역 내에서의 상대 위치 계산
    x_relative = (x_img - plot_x_min) / (plot_x_max - plot_x_min)
    y_relative = (y_img_flipped - plot_y_min) / (plot_y_max - plot_y_min)
    
    # 실제 데이터 좌표로 변환
    x_data = x_min + x_relative * (x_max - x_min)
    y_data = y_min + y_relative * (y_max - y_min)
    
    return x_data, y_data


def extract_point_data_from_image(image_file, points):
    """
    이미지에서 점 데이터를 추출
    """
    try:
        image = Image.open(image_file)
        img_array = np.array(image)
        
        if points is None or len(points) == 0:
            return None
        
        # 점에서 데이터 포인트 추출 (간단한 예시)
        # 실제로는 좌표 변환 및 축 스케일 추출이 필요
        st.info("💡 점 데이터 추출: Prism 스타일 interpolation을 수행합니다.")
        
        # 여기서는 수동 입력으로 대체
        return None
        
    except Exception as e:
        st.error(f"점 데이터 추출 오류: {e}")
        return None


def manual_data_entry(data_type="점"):
    """
    수동으로 데이터 포인트를 입력받는 함수
    data_type: "점" 또는 "선"
    """
    st.subheader(f"📝 수동 데이터 입력 ({data_type} 데이터)")
    
    num_curves = st.number_input("곡선 개수 (농도 조건 수)", min_value=1, max_value=20, value=1)
    
    all_curves_data = {}
    
    for curve_idx in range(num_curves):
        with st.expander(f"곡선 {curve_idx + 1} (농도 조건)", expanded=(curve_idx == 0)):
            conc_name = st.text_input(f"농도 이름 {curve_idx + 1}", value=f"{curve_idx + 1} ug/mL", key=f"conc_{curve_idx}")
            conc_value = st.number_input(f"농도 값 (ug/mL) {curve_idx + 1}", value=float(curve_idx + 1), step=0.1, key=f"conc_val_{curve_idx}")
            
            num_points = st.number_input(f"데이터 포인트 개수 {curve_idx + 1}", min_value=2, max_value=100, value=10, key=f"num_{curve_idx}")
            
            data_points = []
            cols = st.columns(2)
            
            with cols[0]:
                st.write("**시간 (min)**")
            with cols[1]:
                st.write("**RFU 값**")
            
            for i in range(num_points):
                cols = st.columns(2)
                with cols[0]:
                    time_val = st.number_input(f"시간 {i+1}", key=f"time_{curve_idx}_{i}", value=float(i*5), step=0.1)
                with cols[1]:
                    rfu_val = st.number_input(f"RFU {i+1}", key=f"rfu_{curve_idx}_{i}", value=float(100+i*10), step=0.1)
                
                data_points.append({'Time_min': time_val, 'RFU': rfu_val})
            
            all_curves_data[conc_name] = {
                'concentration': conc_value,
                'data': data_points
            }
    
    if st.button("데이터 확인", key="confirm_data"):
        return all_curves_data
    
    return None


def data_load_mode(st):
    """Data Load 모드 - CSV 파일 업로드 또는 이미지에서 데이터 추출"""
    
    # 폴더 구조 생성
    os.makedirs("prep_raw_data_mode", exist_ok=True)
    os.makedirs("prep_raw_data_mode/results", exist_ok=True)
    os.makedirs("data_interpolation_mode/results", exist_ok=True)
    
    st.header("📥 Data Load 모드")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.title("⚙️ Data Load 설정")
    
    # CSV/XLSX 파일 업로드
    st.sidebar.subheader("📁 데이터 파일 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "Prep Raw 데이터 파일 업로드 (CSV 또는 XLSX)",
        type=['csv', 'xlsx'],
        help="prep_raw.csv/xlsx 형식: 시간, 농도별 값, SD, 복제수 (3개 컬럼씩)"
    )
    
    # 샘플 데이터 다운로드
    try:
        with open("mode_prep_raw_data/raw.csv", "rb") as f:
            sample_bytes = f.read()
        st.sidebar.download_button(
            label="샘플 raw.csv 다운로드",
            data=sample_bytes,
            file_name="raw_sample.csv",
            mime="text/csv"
        )
    except Exception:
        pass
    
    # 데이터 로드
    if uploaded_file is not None:
        # 업로드된 파일을 임시로 저장하고 읽기
        import tempfile
        
        # 파일 확장자 확인
        file_extension = uploaded_file.name.split('.')[-1].lower()
        suffix = f'.{file_extension}'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            raw_data = read_raw_data(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return
    else:
        # 기본 샘플 데이터 사용
        from pathlib import Path
        
        # 여러 경로 시도 (Streamlit 실행 경로 문제 대응)
        possible_paths = [
            'mode_prep_raw_data/raw.csv',  # 현재 작업 디렉토리 기준
            str(Path(__file__).parent.parent / 'mode_prep_raw_data' / 'raw.csv'),  # 스크립트 기준
        ]
        
        raw_data = None
        used_path = None
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    raw_data = read_raw_data(path)
                    used_path = path
                    break
            except Exception:
                continue
        
        if raw_data is None:
            # 마지막 시도: 현재 작업 디렉토리에서 직접 찾기
            try:
                raw_data = read_raw_data('mode_prep_raw_data/raw.csv')
                st.sidebar.info("mode_prep_raw_data/raw.csv 사용 중")
            except Exception as e:
                st.error(f"데이터 파일을 찾을 수 없습니다. CSV 또는 XLSX 파일을 업로드해주세요.\n오류: {str(e)}")
                st.stop()
        else:
            st.sidebar.info("mode_prep_raw_data/raw.csv 사용 중")
    
    # 데이터 미리보기
    st.subheader("📋 데이터 미리보기")
    
    # 반응 시간 계산 (최대값)
    all_times = [time_val for data in raw_data.values() for time_val in data['time']] if raw_data else []
    reaction_time = f"{max(all_times):.0f} min" if all_times else "N/A"
    
    # N 값 읽기
    try:
        if uploaded_file is not None:
            uploaded_file.seek(0)
            first_line = uploaded_file.readline().decode('utf-8')
            second_line = uploaded_file.readline().decode('utf-8')
            third_line = uploaded_file.readline().decode('utf-8')
            n_value = int(third_line.split('\t')[3]) if len(third_line.split('\t')) > 3 else 50
            uploaded_file.seek(0)
        else:
            with open('mode_prep_raw_data/raw.csv', 'r', encoding='utf-8') as f:
                f.readline()
                f.readline()
                third_line = f.readline()
                n_value = int(third_line.split('\t')[3]) if len(third_line.split('\t')) > 3 else 50
    except:
        n_value = 50
    
    # raw_data가 없으면 에러 메시지 표시
    if not raw_data:
        st.error("데이터를 로드할 수 없습니다. CSV 또는 XLSX 파일을 업로드해주세요.")
        return
    
    # 농도별 데이터 포인트 수 계산 (모든 농도에서 동일)
    sorted_conc = sorted(raw_data.items(), key=lambda x: x[1]['concentration'])
    num_data_points = len(sorted_conc[0][1]['time']) if len(sorted_conc) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("농도 조건 수", len(raw_data))
    with col2:
        st.metric("농도별 데이터 포인트 수", num_data_points)
    with col3:
        st.metric("반응 시간", reaction_time)
    with col4:
        st.metric("N(시험 수)", n_value)
    
    # 농도별 정보 표시
    with st.expander("농도별 데이터 정보", expanded=False):
        if len(sorted_conc) > 0:
            first_data = sorted_conc[0][1]
            times = first_data['time']
            
            detail_data = {'time_min': times}
            for conc_name, data in sorted_conc:
                conc_label = f"{data['concentration']}"
                detail_data[f'{conc_label}_mean'] = data['value']
                if data.get('SD') is not None:
                    detail_data[f'{conc_label}_SD'] = data['SD']
            
            detail_df = pd.DataFrame(detail_data)
            st.dataframe(detail_df, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("데이터가 없습니다.")
    
    # 실험 조건 선택
    st.sidebar.subheader("🔬 실험 조건 설정")
    experiment_type = st.sidebar.radio(
        "실험 조건",
        ["Substrate 농도 변화 (표준 MM)", "Enzyme 농도 변화 (Substrate 고정)"],
        help="Substrate 농도 변화: 표준 MM 적용 가능 | Enzyme 농도 변화: 표준 MM 적용 불가, 선형 관계"
    )
    
    if experiment_type == "Enzyme 농도 변화 (Substrate 고정)":
        st.sidebar.warning("""
        ⚠️ **주의: 표준 Michaelis-Menten 모델이 아닙니다**
        
        - Substrate 고정 + Enzyme 농도 변화 실험
        - v는 [E]에 대해 **선형(linear)** 관계
        - **Km을 구할 수 없음** (기질 농도 gradient 필요)
        - 구할 수 있는 파라미터: **kcat** 또는 **kcat/Km** (제한적)
        """)
    
    # Michaelis-Menten 모델 실행 버튼
    if st.button("🚀 Michaelis-Menten Model 실행", type="primary"):
            with st.spinner("Michaelis-Menten 모델 피팅 진행 중..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. 각 농도별 시간 경과 곡선 피팅
                status_text.text("1️⃣ 각 농도별 시간 경과 곡선 피팅 중...")
                progress_bar.progress(0.2)
                
                mm_results = {}
                all_fit_data = []
                
                for conc_name, data in raw_data.items():
                    times = data['time']
                    values = data['value']
                    
                    # 초기 속도 계산 (선형 구간 분석)
                    params, fit_values, r_sq = fit_time_course(times, values, model='linear')
                    
                    # 초기 속도 파라미터 추출
                    v0 = params['v0']  # 초기 속도
                    F0 = params['F0']  # 초기 형광값
                    Fmax = params['Fmax']  # 최대 형광값
                    
                    mm_results[conc_name] = {
                        'concentration': data['concentration'],
                        'v0': v0,
                        'F0': F0,
                        'Fmax': Fmax,
                        'R_squared': r_sq,
                        'linear_fraction': params['linear_fraction']
                    }
                    
                    # Fit curve 데이터 저장 (선형 구간만)
                    valid_mask = ~np.isnan(fit_values)
                    # 농도 단위 자동 감지
                    conc_unit_col = 'Concentration [μM]' if 'μM' in conc_name or 'uM' in conc_name else 'Concentration [ug/mL]'
                    
                    for t, val, fit_val in zip(times[valid_mask], values[valid_mask], fit_values[valid_mask]):
                        fit_row = {
                            'Concentration': conc_name,
                            'Time_min': t,
                            'Observed_Value': val,
                            'Fit_Value': fit_val,
                            'Residual': val - fit_val
                        }
                        fit_row[conc_unit_col] = data['concentration']
                        all_fit_data.append(fit_row)
                
                progress_bar.progress(0.4)
                
                # 2. Interpolation 범위 계산
                status_text.text("2️⃣ 보간 범위 계산 중...")
                
                all_times = [time_val for data in raw_data.values() for time_val in data['time']]
                x_data_min = min(all_times)
                x_data_max = max(all_times)
                # 원본 데이터 범위만 사용 (Prism 확장 범위 사용 안 함)
                x_range_min = x_data_min
                x_range_max = x_data_max
                
                # 보간 포인트 개수 설정 (고정값 사용)
                n_points = 1000  # 기본값으로 고정
                
                # 고밀도 보간 포인트 생성
                x_interp = np.linspace(x_range_min, x_range_max, n_points + 1)
                
                progress_bar.progress(0.6)
                
                # 3. Interpolation 수행
                status_text.text("3️⃣ 보간 곡선 생성 중...")
                
                all_interp_data = []
                # 농도 단위 자동 감지 (첫 번째 농도 이름에서 확인)
                first_conc_name = list(mm_results.keys())[0] if mm_results else ""
                conc_unit_col = 'Concentration [μM]' if 'μM' in first_conc_name or 'uM' in first_conc_name else 'Concentration [ug/mL]'
                
                for conc_name, params in mm_results.items():
                    v0 = params['v0']
                    F0 = params['F0']
                    Fmax = params['Fmax']
                    
                    # 선형 피팅으로 보간 (v0 = 기울기)
                    # F(t) = F0 + v0 * t
                    y_interp = F0 + v0 * x_interp
                    # Fmax를 넘지 않도록 제한
                    y_interp = np.clip(y_interp, F0, Fmax)
                    
                    for x, y in zip(x_interp, y_interp):
                        interp_row = {
                            'Concentration': conc_name,
                            'Time_min': x,
                            'RFU_Interpolated': y
                        }
                        interp_row[conc_unit_col] = params['concentration']
                        all_interp_data.append(interp_row)
                
                interp_df = pd.DataFrame(all_interp_data)
                
                progress_bar.progress(0.7)
                
                # 4. v₀ vs 농도 피팅 (실험 조건에 따라 다름)
                if experiment_type == "Substrate 농도 변화 (표준 MM)":
                    status_text.text("4️⃣ v₀ vs [S] Michaelis-Menten 피팅 중...")
                    
                    # 농도와 초기 속도 데이터 수집
                    concentrations = [params['concentration'] for params in sorted(mm_results.values(), 
                                                                                  key=lambda x: x['concentration'])]
                    v0_values = [params['v0'] for params in sorted(mm_results.values(), 
                                                                  key=lambda x: x['concentration'])]
                    
                    # MM calibration curve 피팅: v₀ = Vmax * [S] / (Km + [S])
                    if len(concentrations) >= 2 and len(v0_values) >= 2:
                        try:
                            cal_params, cal_fit_values, cal_equation = fit_calibration_curve(concentrations, v0_values)
                            Vmax = cal_params['Vmax_cal']
                            Km = cal_params['Km_cal']
                            mm_r_squared = cal_params['R_squared']
                            
                            # kcat 계산 (enzyme 농도 필요)
                            kcat = None
                            mm_fit_success = True
                        except Exception as e:
                            st.warning(f"⚠️ MM 피팅 실패: {e}")
                            Vmax = None
                            Km = None
                            kcat = None
                            mm_r_squared = 0
                            cal_equation = "피팅 실패"
                            mm_fit_success = False
                    else:
                        Vmax = None
                        Km = None
                        kcat = None
                        mm_r_squared = 0
                        cal_equation = "데이터 부족 (최소 2개 농도 필요)"
                        mm_fit_success = False
                
                else:  # Enzyme 농도 변화 (Substrate 고정)
                    status_text.text("4️⃣ v₀ vs [E] 선형 피팅 중... (표준 MM 아님)")
                    
                    # 농도와 초기 속도 데이터 수집
                    concentrations = [params['concentration'] for params in sorted(mm_results.values(), 
                                                                                  key=lambda x: x['concentration'])]
                    v0_values = [params['v0'] for params in sorted(mm_results.values(), 
                                                                  key=lambda x: x['concentration'])]
                    
                    # 선형 피팅: v = kcat * [E] * [S] / (Km + [S])
                    # Substrate 고정이므로 slope = kcat * [S] / (Km + [S])
                    if len(concentrations) >= 2 and len(v0_values) >= 2:
                        try:
                            # 선형 회귀
                            coeffs = np.polyfit(concentrations, v0_values, 1)
                            slope = coeffs[0]  # kcat * [S] / (Km + [S])
                            intercept = coeffs[1]
                            
                            # 피팅된 값
                            v0_fitted = np.polyval(coeffs, concentrations)
                            
                            # R² 계산
                            ss_res = np.sum((v0_values - v0_fitted) ** 2)
                            ss_tot = np.sum((v0_values - np.mean(v0_values)) ** 2)
                            mm_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                            
                            # Substrate 고정 조건에서는 Km을 구할 수 없음
                            Vmax = None  # Vmax는 [E] 고정 조건에서만 정의됨
                            Km = None  # 구할 수 없음
                            kcat = None  # 단독으로 구할 수 없음 (kcat/Km만 가능)
                            
                            # slope = kcat * [S] / (Km + [S])
                            # Substrate 농도가 알려져 있으면 kcat/Km을 추정할 수 있음 (희석 조건)
                            cal_equation = f"v₀ = {slope:.4f} * [E] + {intercept:.4f} (선형)"
                            mm_fit_success = True
                            
                            # 경고 메시지
                            st.warning("""
                            ⚠️ **Substrate 고정 + Enzyme 농도 변화 실험**
                            
                            - v는 [E]에 대해 **선형(linear)** 관계입니다
                            - **Km을 구할 수 없습니다** (기질 농도 gradient 필요)
                            - **Vmax를 구할 수 없습니다** (표준 MM 정의에선 [E] 고정 필요)
                            - 구할 수 있는 파라미터:
                              - **slope = kcat × [S] / (Km + [S])**
                              - Substrate 농도가 매우 낮으면: **slope ≈ kcat/Km × [S]**
                            """)
                        except Exception as e:
                            st.warning(f"⚠️ 선형 피팅 실패: {e}")
                            Vmax = None
                            Km = None
                            kcat = None
                            mm_r_squared = 0
                            cal_equation = "피팅 실패"
                            mm_fit_success = False
                    else:
                        Vmax = None
                        Km = None
                        kcat = None
                        mm_r_squared = 0
                        cal_equation = "데이터 부족 (최소 2개 농도 필요)"
                        mm_fit_success = False
                
                progress_bar.progress(0.85)
                
                # 5. 결과 저장
                status_text.text("5️⃣ 결과 저장 중...")
                
                # 초기 속도 Results 저장 (MM 파라미터 포함)
                # 농도 단위 자동 감지 (첫 번째 농도 이름에서 확인)
                first_conc_name = list(mm_results.keys())[0] if mm_results else ""
                conc_unit_col = 'Concentration [μM]' if 'μM' in first_conc_name or 'uM' in first_conc_name else 'Concentration [ug/mL]'
                
                results_data = []
                for conc_name, params in sorted(mm_results.items(), key=lambda x: x[1]['concentration']):
                    eq = f"v0 = {params['v0']:.2f} (선형 구간 기울기)"
                    result_row = {
                        'Concentration': conc_name,
                        'v0': params['v0'],
                        'F0': params['F0'],
                        'Fmax': params['Fmax'],
                        'R_squared': params['R_squared'],
                        'linear_fraction': params['linear_fraction'],
                        'Equation': eq
                    }
                    result_row[conc_unit_col] = params['concentration']
                    results_data.append(result_row)
                
                mm_results_df = pd.DataFrame(results_data)
                
                # 저장된 xlsx 파일에서 enzyme 농도 읽기 시도 (kcat 계산용)
                enzyme_conc = None
                try:
                    xlsx_path = 'Michaelis-Menten_calibration_results.xlsx'
                    if os.path.exists(xlsx_path):
                        df_mm_read = pd.read_excel(xlsx_path, sheet_name='MM Results', engine='openpyxl')
                        # enzyme 농도 컬럼 찾기 (다양한 이름 시도)
                        enzyme_conc_col = None
                        for col in ['Enzyme [ug/mL]', 'Enzyme_ug/mL', 'enzyme_ug/mL', '[E] (ug/mL)', 'E_conc', 'Enzyme']:
                            if col in df_mm_read.columns:
                                enzyme_conc_col = col
                                break
                        
                        if enzyme_conc_col is not None:
                            # 첫 번째 유효한 enzyme 농도 값 사용
                            enzyme_conc_values = df_mm_read[enzyme_conc_col].dropna()
                            if len(enzyme_conc_values) > 0:
                                enzyme_conc = float(enzyme_conc_values.iloc[0])
                except Exception as e:
                    # enzyme 농도 읽기 실패해도 계속 진행
                    pass
                
                # kcat 계산: kcat = Vmax / [E]_T
                if mm_fit_success and Vmax is not None and enzyme_conc is not None and enzyme_conc > 0:
                    kcat = Vmax / enzyme_conc
                else:
                    kcat = None
                
                # MM 피팅 결과를 별도로 저장
                mm_fit_results = {
                    'Vmax': Vmax,
                    'Km': Km,
                    'kcat': kcat,
                    'enzyme_conc': enzyme_conc,
                    'R_squared': mm_r_squared,
                    'equation': cal_equation,
                    'fit_success': mm_fit_success,
                    'experiment_type': experiment_type,
                    'slope': None  # Enzyme 농도 변화인 경우 slope 저장
                }
                
                # Enzyme 농도 변화인 경우 slope 저장
                if experiment_type == "Enzyme 농도 변화 (Substrate 고정)" and mm_fit_success:
                    concentrations = [params['concentration'] for params in sorted(mm_results.values(), 
                                                                                  key=lambda x: x['concentration'])]
                    v0_values = [params['v0'] for params in sorted(mm_results.values(), 
                                                                  key=lambda x: x['concentration'])]
                    if len(concentrations) >= 2:
                        coeffs = np.polyfit(concentrations, v0_values, 1)
                        mm_fit_results['slope'] = coeffs[0]
                        mm_fit_results['intercept'] = coeffs[1]
                
                try:
                    # Interpolated curves 저장 (CSV)
                    interp_df.to_csv('data_interpolation_mode/results/MM_interpolated_curves.csv', index=False)
                    
                    # MM results 저장 (CSV)
                    mm_results_df.to_csv('prep_raw_data_mode/results/MM_results_detailed.csv', index=False)
                    
                    st.sidebar.success("✅ 결과 파일이 저장되었습니다!")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ 파일 저장 중 오류: {e}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Michaelis-Menten 모델 피팅 완료!")
                
                # Session state에 저장
                st.session_state['interpolation_results'] = {
                    'interp_df': interp_df,
                    'mm_results_df': mm_results_df,
                    'mm_fit_results': mm_fit_results,
                    'x_range_min': x_range_min,
                    'x_range_max': x_range_max,
                    'x_data_min': x_data_min,
                    'x_data_max': x_data_max,
                    'raw_data': raw_data,
                    'v0_vs_concentration': {
                        'concentrations': concentrations,
                        'v0_values': v0_values
                    },
                    'experiment_type': experiment_type
                }
    
    # 결과 표시
    if 'interpolation_results' in st.session_state:
            results = st.session_state['interpolation_results']
            
            st.markdown("---")
            st.subheader("📊 Michaelis-Menten 모델 결과")
            
            # MM 피팅 결과 표시 (실험 조건에 따라 다름)
            if 'mm_fit_results' in results and results['mm_fit_results']['fit_success']:
                mm_fit = results['mm_fit_results']
                exp_type = mm_fit.get('experiment_type', 'Substrate 농도 변화 (표준 MM)')
                
                if exp_type == "Substrate 농도 변화 (표준 MM)":
                    # 표준 MM 결과 표시
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Vmax", f"{mm_fit['Vmax']:.2f}" if mm_fit['Vmax'] is not None else "N/A")
                    with col2:
                        st.metric("Km (μg/mL)", f"{mm_fit['Km']:.4f}" if mm_fit['Km'] is not None else "N/A")
                    with col3:
                        st.metric("kcat", f"{mm_fit['kcat']:.2f}" if mm_fit['kcat'] is not None else "N/A")
                    with col4:
                        st.metric("R²", f"{mm_fit['R_squared']:.4f}")
                    
                    st.info(f"**MM 방정식:** {mm_fit['equation']}")
                else:
                    # Enzyme 농도 변화 결과 표시
                    st.warning("⚠️ **Substrate 고정 + Enzyme 농도 변화 실험** (표준 MM 아님)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        slope = mm_fit.get('slope', None)
                        st.metric("Slope (v₀ vs [E])", f"{slope:.4f}" if slope is not None else "N/A",
                                 help="slope = kcat × [S] / (Km + [S])")
                    with col2:
                        intercept = mm_fit.get('intercept', None)
                        st.metric("Intercept", f"{intercept:.4f}" if intercept is not None else "N/A")
                    with col3:
                        st.metric("R²", f"{mm_fit['R_squared']:.4f}")
                    
                    st.info(f"**선형 방정식:** {mm_fit['equation']}")
                    st.info("""
                    📌 **구할 수 있는 파라미터:**
                    - **Slope**: kcat × [S] / (Km + [S])
                    - Substrate 농도가 매우 낮으면: slope ≈ kcat/Km × [S]
                    
                    ❌ **구할 수 없는 파라미터:**
                    - **Km**: 기질 농도 gradient 필요
                    - **Vmax**: 표준 MM 정의에선 [E] 고정 필요
                    - **kcat**: 단독으로 구할 수 없음 (kcat/Km만 가능)
                    """)
            elif 'mm_fit_results' in results:
                st.warning("⚠️ MM 피팅 실패 또는 데이터 부족")
            
            # 탭 구성
            tabs = ["📈 Time-Fluorescence Curves", "📊 v₀ vs [S] MM Fit", "📋 Data Table"]
            tab_objects = st.tabs(tabs)
            
            # Tab 1: Time-Fluorescence 그래프
            with tab_objects[0]:
                st.subheader("Time-Fluorescence Curves")
                
                fig = go.Figure()
                colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                
                # 농도 순서대로 정렬
                conc_col = None
                for col in ['Concentration [μM]', 'Concentration [ug/mL]']:
                    if col in results['mm_results_df'].columns:
                        conc_col = col
                        break
                
                if conc_col:
                    conc_order = results['mm_results_df'].sort_values(conc_col)['Concentration'].tolist()
                else:
                    conc_order = results['mm_results_df']['Concentration'].tolist()
                
                x_data_min = results['x_data_min']
                x_data_max = results['x_data_max']
                
                for idx, conc_name in enumerate(conc_order):
                    color = colors[idx % len(colors)]
                    
                    # 보간 곡선
                    subset = results['interp_df'][results['interp_df']['Concentration'] == conc_name]
                    
                    if len(subset) > 0:
                        fig.add_trace(go.Scatter(
                            x=subset['Time_min'],
                            y=subset['RFU_Interpolated'],
                            mode='lines',
                            name=conc_name,
                            line=dict(color=color, width=2.5),
                            legendgroup=conc_name,
                            showlegend=True
                        ))
                
                fig.update_layout(
                    xaxis_title='Time (min)',
                    yaxis_title='RFU',
                    height=700,
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    legend=dict(
                        orientation="v",
                        yanchor="bottom",
                        y=0.05,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        font=dict(color="white")
                    )
                )
                
                # 원본 데이터 시간 범위로 제한
                fig.update_xaxes(range=[results['x_data_min'], results['x_data_max']])
                fig.update_yaxes(rangemode='tozero')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: v₀ vs 농도 그래프 (실험 조건에 따라 다름)
            with tab_objects[1]:
                if 'v0_vs_concentration' in results and 'mm_fit_results' in results:
                    v0_data = results['v0_vs_concentration']
                    mm_fit = results['mm_fit_results']
                    exp_type = mm_fit.get('experiment_type', 'Substrate 농도 변화 (표준 MM)')
                    
                    fig_v0 = go.Figure()
                    
                    # 실험 데이터 포인트
                    fig_v0.add_trace(go.Scatter(
                        x=v0_data['concentrations'],
                        y=v0_data['v0_values'],
                        mode='markers',
                        name='Experimental v₀',
                        marker=dict(size=10, color='red', line=dict(width=2, color='black'))
                    ))
                    
                    if exp_type == "Substrate 농도 변화 (표준 MM)":
                        st.subheader("v₀ vs [S] Michaelis-Menten Fit")
                        
                        # MM 피팅 곡선
                        if mm_fit['fit_success'] and mm_fit['Vmax'] is not None and mm_fit['Km'] is not None:
                            conc_min = min(v0_data['concentrations'])
                            conc_max = max(v0_data['concentrations'])
                            conc_range = np.linspace(conc_min * 0.5, conc_max * 1.5, 200)
                            v0_fitted = michaelis_menten_calibration(conc_range, mm_fit['Vmax'], mm_fit['Km'])
                            
                            fig_v0.add_trace(go.Scatter(
                                x=conc_range,
                                y=v0_fitted,
                                mode='lines',
                                name=f'MM Fit: {mm_fit["equation"]}',
                                line=dict(width=2.5, color='blue')
                            ))
                            
                            # 통계 정보
                            stats_text = f"Vmax = {mm_fit['Vmax']:.2f}<br>"
                            stats_text += f"Km = {mm_fit['Km']:.4f} μg/mL<br>"
                            stats_text += f"R² = {mm_fit['R_squared']:.4f}"
                            
                            fig_v0.add_annotation(
                                xref="paper", yref="paper",
                                x=0.05, y=0.95,
                                xanchor='left', yanchor='top',
                                text=stats_text,
                                showarrow=False,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="blue",
                                borderwidth=2,
                                font=dict(size=11)
                            )
                        
                        fig_v0.update_layout(
                            title='Initial Velocity (v₀) vs Substrate Concentration [S]',
                            xaxis_title='[S] (μg/mL)',
                            yaxis_title='Initial Velocity v₀ (Fluorescence Units / Time)',
                            template='plotly_white',
                            height=600,
                            hovermode='x unified'
                        )
                    else:
                        st.subheader("v₀ vs [E] Linear Fit (Substrate 고정)")
                        st.warning("⚠️ 표준 Michaelis-Menten 모델이 아닙니다. v는 [E]에 대해 선형 관계입니다.")
                        
                        # 선형 피팅 곡선
                        if mm_fit['fit_success'] and mm_fit.get('slope') is not None:
                            conc_min = min(v0_data['concentrations'])
                            conc_max = max(v0_data['concentrations'])
                            conc_range = np.linspace(conc_min * 0.5, conc_max * 1.5, 200)
                            slope = mm_fit['slope']
                            intercept = mm_fit.get('intercept', 0)
                            v0_fitted = slope * conc_range + intercept
                            
                            fig_v0.add_trace(go.Scatter(
                                x=conc_range,
                                y=v0_fitted,
                                mode='lines',
                                name=f'Linear Fit: {mm_fit["equation"]}',
                                line=dict(width=2.5, color='blue', dash='dash')
                            ))
                            
                            # 통계 정보
                            stats_text = f"Slope = {slope:.4f}<br>"
                            stats_text += f"Intercept = {intercept:.4f}<br>"
                            stats_text += f"R² = {mm_fit['R_squared']:.4f}<br>"
                            stats_text += "<br><b>⚠️ Km을 구할 수 없음</b>"
                            
                            fig_v0.add_annotation(
                                xref="paper", yref="paper",
                                x=0.05, y=0.95,
                                xanchor='left', yanchor='top',
                                text=stats_text,
                                showarrow=False,
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="orange",
                                borderwidth=2,
                                font=dict(size=11)
                            )
                        
                        fig_v0.update_layout(
                            title='Initial Velocity (v₀) vs Enzyme Concentration [E] (Substrate 고정)',
                            xaxis_title='[E] (μg/mL 또는 μM)',
                            yaxis_title='Initial Velocity v₀ (Fluorescence Units / Time)',
                            template='plotly_white',
                            height=600,
                            hovermode='x unified'
                        )
                    
                    st.plotly_chart(fig_v0, use_container_width=True)
                else:
                    st.warning("v₀ vs 농도 데이터가 없습니다.")
            
            # Tab 3: 데이터 테이블
            with tab_objects[2]:
                st.subheader("상세 파라미터")
                
                # 상세 파라미터 테이블
                detail_cols = ['Concentration [μM]', 'Concentration [ug/mL]', 'v0', 'F0', 'Fmax', 'R_squared', 'linear_fraction', 'Equation']
                available_cols = [col for col in detail_cols if col in results['mm_results_df'].columns]
                st.dataframe(results['mm_results_df'][available_cols], use_container_width=True, hide_index=True)
                
                # 파일 다운로드 버튼
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                # MM Results CSV 다운로드
                with col1:
                    mm_results_csv = results['mm_results_df'][available_cols].to_csv(index=False)
                    st.download_button(
                        label="📥 MM Results 다운로드 (CSV)",
                        data=mm_results_csv,
                        file_name="MM_Results.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="MM Results 시트의 데이터를 CSV 파일로 다운로드합니다."
                    )
                
                # XLSX 다운로드 버튼 및 자동 저장
                with col2:
                    try:
                        from io import BytesIO
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results['mm_results_df'][available_cols].to_excel(writer, sheet_name='MM Results', index=False)
                            results['interp_df'].to_excel(writer, sheet_name='Michaelis-Menten Curves', index=False)
                        output.seek(0)
                        xlsx_data = output.getvalue()
                        
                        # XLSX 파일 자동 저장 (Analysis 모드에서 자동 로드용)
                        try:
                            with open('Michaelis-Menten_calibration_results.xlsx', 'wb') as f:
                                f.write(xlsx_data)
                        except Exception as save_err:
                            st.sidebar.warning(f"⚠️ XLSX 파일 자동 저장 실패: {save_err}")
                        
                        st.download_button(
                            label="📥 전체 결과 다운로드 (XLSX)",
                            data=xlsx_data,
                            file_name="Michaelis-Menten_calibration_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            help="MM Results와 Michaelis-Menten Curves 시트를 포함한 전체 엑셀 파일입니다."
                        )
                    except Exception as e:
                        st.warning(f"XLSX 다운로드 준비 중 오류: {e}")

