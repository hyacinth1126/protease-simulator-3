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
    
    # 데이터 소스 선택
    st.sidebar.subheader("📁 데이터 소스 선택")
    data_source = st.sidebar.radio(
        "데이터 입력 방법",
        ["CSV 파일 업로드", "이미지 파일 업로드"],
        help="CSV 파일: prep_raw.csv 형식 직접 업로드 | 이미지 파일: 그래프 이미지에서 데이터 추출"
    )
    
    if data_source == "CSV 파일 업로드":
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
        all_times = [time_val for data in raw_data.values() for time_val in data['time']]
        reaction_time = f"{max(all_times):.0f} min"
        
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
                    for t, val, fit_val in zip(times[valid_mask], values[valid_mask], fit_values[valid_mask]):
                        all_fit_data.append({
                            'Concentration': conc_name,
                            'Concentration [ug/mL]': data['concentration'],
                            'Time_min': t,
                            'Observed_Value': val,
                            'Fit_Value': fit_val,
                            'Residual': val - fit_val
                        })
                
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
                        all_interp_data.append({
                            'Concentration': conc_name,
                            'Concentration [ug/mL]': params['concentration'],
                            'Time_min': x,
                            'RFU_Interpolated': y
                        })
                
                interp_df = pd.DataFrame(all_interp_data)
                
                progress_bar.progress(0.7)
                
                # 4. v₀ vs [S]에 Michaelis-Menten 피팅
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
                        
                        # kcat 계산 (enzyme 농도 필요 - 일단 None으로 설정, 나중에 사용자 입력 받을 수 있음)
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
                
                progress_bar.progress(0.85)
                
                # 5. 결과 저장
                status_text.text("5️⃣ 결과 저장 중...")
                
                # 초기 속도 Results 저장 (MM 파라미터 포함)
                results_data = []
                for conc_name, params in sorted(mm_results.items(), key=lambda x: x[1]['concentration']):
                    eq = f"v0 = {params['v0']:.2f} (선형 구간 기울기)"
                    results_data.append({
                        'Concentration': conc_name,
                        'Concentration [ug/mL]': params['concentration'],
                        'v0': params['v0'],
                        'F0': params['F0'],
                        'Fmax': params['Fmax'],
                        'R_squared': params['R_squared'],
                        'linear_fraction': params['linear_fraction'],
                        'Equation': eq
                    })
                
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
                    'fit_success': mm_fit_success
                }
                
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
                    }
                }
        
        # 결과 표시
        if 'interpolation_results' in st.session_state:
            results = st.session_state['interpolation_results']
            
            st.markdown("---")
            st.subheader("📊 Michaelis-Menten 모델 결과")
            
            # MM 피팅 결과 표시 (Vmax, Km, kcat)
            if 'mm_fit_results' in results and results['mm_fit_results']['fit_success']:
                mm_fit = results['mm_fit_results']
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
                if 'Concentration [ug/mL]' in results['mm_results_df'].columns:
                    conc_order = results['mm_results_df'].sort_values('Concentration [ug/mL]')['Concentration'].tolist()
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
            
            # Tab 2: v₀ vs [S] MM Fit 그래프
            with tab_objects[1]:
                st.subheader("v₀ vs [S] Michaelis-Menten Fit")
                
                if 'v0_vs_concentration' in results and 'mm_fit_results' in results:
                    v0_data = results['v0_vs_concentration']
                    mm_fit = results['mm_fit_results']
                    
                    fig_v0 = go.Figure()
                    
                    # 실험 데이터 포인트
                    fig_v0.add_trace(go.Scatter(
                        x=v0_data['concentrations'],
                        y=v0_data['v0_values'],
                        mode='markers',
                        name='Experimental v₀',
                        marker=dict(size=10, color='red', line=dict(width=2, color='black'))
                    ))
                    
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
                    
                    st.plotly_chart(fig_v0, use_container_width=True)
                else:
                    st.warning("v₀ vs [S] 데이터가 없습니다.")
            
            # Tab 3: 데이터 테이블
            with tab_objects[2]:
                st.subheader("상세 파라미터")
                
                # 상세 파라미터 테이블
                detail_cols = ['Concentration [ug/mL]', 'v0', 'F0', 'Fmax', 'R_squared', 'linear_fraction', 'Equation']
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
    
    else:  # 이미지 파일 업로드
        st.sidebar.subheader("📁 이미지 파일 업로드")
        uploaded_image = st.sidebar.file_uploader(
            "그래프 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            help="그래프 이미지에서 선 또는 점 데이터를 추출합니다"
        )
        
        # 샘플 이미지 다운로드
        try:
            with open("raw.png", "rb") as f:
                sample_bytes = f.read()
            st.sidebar.download_button(
                label="샘플 raw.png 다운로드",
                data=sample_bytes,
                file_name="raw_sample.png",
                mime="image/png"
            )
        except Exception:
            pass
        
        # 이미지 로드 (업로드된 파일 또는 기본 샘플)
        if uploaded_image is not None:
            # 업로드된 이미지 사용
            image = Image.open(uploaded_image)
            img_array = np.array(image)
            st.image(image, caption="업로드된 이미지")
        else:
            # 기본 샘플 이미지 사용
            try:
                from pathlib import Path
                
                # 여러 경로 시도
                possible_paths = [
                    'raw.png',
                    str(Path(__file__).parent.parent / 'raw.png'),
                ]
                
                image = None
                for path in possible_paths:
                    try:
                        if os.path.exists(path):
                            image = Image.open(path)
                            break
                    except Exception:
                        continue
                
                if image is None:
                    # 마지막 시도
                    image = Image.open('raw.png')
                
                img_array = np.array(image)
                st.image(image, caption="샘플 이미지 (raw.png)")
                st.sidebar.info("raw.png 사용 중")
            except FileNotFoundError:
                st.error("이미지 파일을 찾을 수 없습니다. 이미지 파일을 업로드하거나 raw.png 파일을 프로젝트 루트에 배치해주세요.")
                st.stop()
            except Exception as e:
                st.error(f"이미지 파일 로드 오류: {e}")
                st.stop()
        
        if image is not None:
            
            # 이미지에서 데이터 추출 시도
            st.subheader("📊 이미지에서 데이터 추출")
            
            # 그래프 타입 선택
            graph_type = st.radio(
                "그래프 타입",
                ["선/점선 그래프", "점 그래프"],
                help="선/점선: Exponential Association 모델로 fitting | 점: Prism 스타일 interpolation"
            )
            
            # 자동 감지 시도
            lines, points = None, None
            if CV2_AVAILABLE:
                lines, points = detect_lines_and_points(img_array)
                if lines is not None and len(lines) > 0:
                    st.info(f"✅ {len(lines)}개의 선이 감지되었습니다.")
                if points is not None and len(points) > 0:
                    st.info(f"✅ {len(points)}개의 점이 감지되었습니다.")
            
            # 수동 입력
            if graph_type == "선/점선 그래프":
                st.info("💡 선 데이터: Exponential Association 모델 F(t) = F0 + (Fmax - F0) * [1 - exp(-k*t)]로 fitting합니다.")
                curves_data = manual_data_entry("선")
            else:
                st.info("💡 점 데이터: Prism 스타일 interpolation을 수행합니다.")
                curves_data = manual_data_entry("점")
            
            if curves_data is not None:
                st.success("✅ 데이터 입력 완료!")
                
                # 데이터 미리보기
                with st.expander("입력된 데이터 미리보기", expanded=True):
                    for conc_name, curve_info in curves_data.items():
                        st.write(f"**{conc_name}** (농도: {curve_info['concentration']} ug/mL)")
                        df_preview = pd.DataFrame(curve_info['data'])
                        st.dataframe(df_preview, use_container_width=True, hide_index=True)
                
                # 처리 실행 버튼
                if st.button("🚀 데이터 처리 실행", type="primary"):
                    with st.spinner("데이터 처리 중..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        all_interp_data = []
                        mm_results = {}
                        all_times_list = []  # 전체 시간 범위 계산용
                        
                        # 각 곡선별 처리
                        for idx, (conc_name, curve_info) in enumerate(curves_data.items()):
                            times = np.array([d['Time_min'] for d in curve_info['data']])
                            values = np.array([d['RFU'] for d in curve_info['data']])
                            conc_value = curve_info['concentration']
                            
                            all_times_list.extend(times.tolist())
                            
                            status_text.text(f"처리 중: {conc_name} ({idx+1}/{len(curves_data)})")
                            progress_bar.progress((idx + 0.5) / len(curves_data))
                            
                            if graph_type == "선/점선 그래프":
                                # 선 데이터: 초기 속도 계산 (선형 구간 분석)
                                params, fit_values, r_sq = fit_time_course(times, values, model='linear')
                                
                                v0 = params['v0']
                                F0 = params['F0']
                                Fmax = params['Fmax']
                                
                                mm_results[conc_name] = {
                                    'concentration': conc_value,
                                    'v0': v0,
                                    'F0': F0,
                                    'Fmax': Fmax,
                                    'R_squared': r_sq,
                                    'linear_fraction': params['linear_fraction']
                                }
                                
                                # Interpolation 범위 계산 (개별 곡선)
                                x_data_min_curve = float(np.min(times))
                                x_data_max_curve = float(np.max(times))
                                x_range_min_curve, x_range_max_curve = create_prism_interpolation_range(times)
                                
                                # 고밀도 보간 포인트 생성
                                n_points = 1000
                                x_interp = np.linspace(x_range_min_curve, x_range_max_curve, n_points + 1)
                                
                                # 선형 피팅으로 계산 (선형 구간만)
                                linear_times = times[~np.isnan(fit_values)]
                                linear_values = values[~np.isnan(fit_values)]
                                if len(linear_times) >= 2:
                                    coeffs = np.polyfit(linear_times, linear_values, 1)
                                    y_interp = np.polyval(coeffs, x_interp)
                                else:
                                    y_interp = np.full_like(x_interp, F0)
                                
                            else:
                                # 점 데이터: 초기 속도 계산 (선형 구간 분석)
                                params, fit_values, r_sq = fit_time_course(times, values, model='linear')
                                
                                v0 = params['v0']
                                F0 = params['F0']
                                Fmax = params['Fmax']
                                
                                mm_results[conc_name] = {
                                    'concentration': conc_value,
                                    'v0': v0,
                                    'F0': F0,
                                    'Fmax': Fmax,
                                    'R_squared': r_sq,
                                    'linear_fraction': params['linear_fraction']
                                }
                                
                                # Interpolation 범위 계산 (개별 곡선)
                                x_data_min_curve = float(np.min(times))
                                x_data_max_curve = float(np.max(times))
                                x_range_min_curve, x_range_max_curve = create_prism_interpolation_range(times)
                                
                                # 고밀도 보간 포인트 생성
                                n_points = 1000
                                x_interp = np.linspace(x_range_min_curve, x_range_max_curve, n_points + 1)
                                
                                # 선형 피팅으로 interpolation (선형 구간만)
                                linear_times = times[~np.isnan(fit_values)]
                                linear_values = values[~np.isnan(fit_values)]
                                if len(linear_times) >= 2:
                                    coeffs = np.polyfit(linear_times, linear_values, 1)
                                    y_interp = np.polyval(coeffs, x_interp)
                                else:
                                    y_interp = np.full_like(x_interp, F0)
                            
                            # Interpolated 데이터 저장
                            for x, y in zip(x_interp, y_interp):
                                all_interp_data.append({
                                    'Concentration': conc_name,
                                    'Concentration [ug/mL]': conc_value,
                                    'Time_min': x,
                                    'RFU_Interpolated': y
                                })
                        
                        # 전체 시간 범위 계산
                        all_times_array = np.array(all_times_list)
                        x_data_min = float(np.min(all_times_array))
                        x_data_max = float(np.max(all_times_array))
                        x_range_min, x_range_max = create_prism_interpolation_range(all_times_array)
                        
                        interp_df = pd.DataFrame(all_interp_data)
                        
                        # 초기 속도 Results 저장
                        results_data = []
                        for conc_name, params in sorted(mm_results.items(), key=lambda x: x[1]['concentration']):
                            eq = f"v0 = {params['v0']:.2f} (선형 구간 기울기)"
                            results_data.append({
                                'Concentration': conc_name,
                                'Concentration [ug/mL]': params['concentration'],
                                'v0': params['v0'],
                                'F0': params['F0'],
                                'Fmax': params['Fmax'],
                                'R_squared': params['R_squared'],
                                'linear_fraction': params['linear_fraction'],
                                'Equation': eq
                            })
                        
                        mm_results_df = pd.DataFrame(results_data)
                        
                        # 결과 저장
                        try:
                            interp_df.to_csv('data_interpolation_mode/results/MM_interpolated_curves.csv', index=False)
                            mm_results_df.to_csv('prep_raw_data_mode/results/MM_results_detailed.csv', index=False)
                            st.sidebar.success("✅ 결과 파일이 저장되었습니다!")
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ 파일 저장 중 오류: {e}")
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ 처리 완료!")
                        
                        # Session state에 저장
                        st.session_state['interpolation_results'] = {
                            'interp_df': interp_df,
                            'mm_results_df': mm_results_df,
                            'x_range_min': x_range_min,
                            'x_range_max': x_range_max,
                            'x_data_min': x_data_min,
                            'x_data_max': x_data_max
                        }
                        
                        st.rerun()
                
                # 결과 표시
                if 'interpolation_results' in st.session_state:
                    results = st.session_state['interpolation_results']
                    
                    st.markdown("---")
                    st.subheader("📊 처리 결과")
                    
                    # 탭 구성
                    tabs = ["📈 Interpolated Curves", "📋 Data Table"]
                    tab_objects = st.tabs(tabs)
                    
                    # Tab 1: 그래프
                    with tab_objects[0]:
                        st.subheader("Interpolated Curves")
                        
                        fig = go.Figure()
                        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                        
                        if 'Concentration [ug/mL]' in results['mm_results_df'].columns:
                            conc_order = results['mm_results_df'].sort_values('Concentration [ug/mL]')['Concentration'].tolist()
                        else:
                            conc_order = results['mm_results_df']['Concentration'].tolist()
                        
                        for idx, conc_name in enumerate(conc_order):
                            color = colors[idx % len(colors)]
                            
                            subset = results['interp_df'][results['interp_df']['Concentration'] == conc_name]
                            
                            if len(subset) > 0:
                                fig.add_trace(go.Scatter(
                                    x=subset['Time_min'],
                                    y=subset['RFU_Interpolated'],
                                    mode='lines',
                                    name=conc_name,
                                    line=dict(color=color, width=2.5)
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tab 2: 데이터 테이블
                    with tab_objects[1]:
                        st.subheader("상세 파라미터")
                        
                        # 상세 파라미터 테이블
                        detail_cols = ['Concentration [ug/mL]', 'v0', 'F0', 'Fmax', 'R_squared', 'linear_fraction', 'Equation']
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
        else:
            st.info("👈 이미지 파일을 업로드해주세요.")

