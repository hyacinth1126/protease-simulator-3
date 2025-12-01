import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

from mode_general_analysis.analysis import (
    UnitStandardizer,
    DataNormalizer,
    RegionDivider,
    ModelA_SubstrateDepletion,
    ModelB_EnzymeDeactivation,
    ModelC_MassTransfer,
    ModelD_ConcentrationDependentFmax,
    ModelE_ProductInhibition,
    ModelF_EnzymeSurfaceSequestration
)
from mode_general_analysis.plot import Visualizer
from mode_prep_raw_data.prep import michaelis_menten_calibration


def verbose_callback(message: str, level: str = "info"):
    """Callback function for logging from analysis modules"""
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "debug":
        st.code(message)
    else:
        st.info(message)


def general_analysis_mode(st):
    """모델 시뮬레이션 모드 - 표준 FRET 분석"""
    
    # Sidebar configuration
    st.sidebar.title("⚙️ 설정")
    
    enzyme_mw = st.sidebar.number_input(
        "효소 분자량 (kDa)",
        min_value=1.0,
        max_value=500.0,
        value=56.6,
        step=0.1,
        help="농도 변환을 위해 필요한 효소 분자량을 입력해주세요."
    )
    
    enzyme_name = st.sidebar.text_input(
        "효소 이름 (선택사항)",
        value="Kgp",
        placeholder="enzyme",
        help="그래프 범례에 표시될 효소 이름 (비워두면 'enzyme' 표시)"
    )
    if enzyme_name.strip() == "":
        enzyme_name = "enzyme"
    
    substrate_name = st.sidebar.text_input(
        "기질 이름 (선택사항)",
        value="Dabcyl-HEK-K(FITC)-C",
        placeholder="substrate",
        help="그래프 범례에 표시될 기질 이름 (비워두면 'substrate' 표시)"
    )
    if substrate_name.strip() == "":
        substrate_name = "substrate"
    # 구분선 후 데이터 소스 섹션
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 데이터 소스")
    
    uploaded_file = st.sidebar.file_uploader(
        "CSV/XLSX 파일 업로드 (Fitted Curves)",
        type=['csv', 'xlsx'],
        help="Data Load 모드에서 생성된 결과 파일 (CSV 또는 XLSX): XLSX의 경우 'Michaelis-Menten Curves' 시트 사용"
    )
    
    # Fitted Curves 샘플 다운로드 (Data Load 모드 결과)
    try:
        with open("data_interpolation_mode/results/MM_interpolated_curves.csv", "rb") as f:
            sample_bytes = f.read()
        st.sidebar.download_button(
            label="📥 Data Load 결과 CSV 다운로드",
            data=sample_bytes,
            file_name="MM_interpolated_curves.csv",
            mime="text/csv",
            help="Data Load 모드에서 생성된 결과 CSV 파일"
        )
    except Exception:
        pass
    
    # Step 1: Load Fitted Curves data (원본 데이터 플롯용)
    df_fitted = None
    rfu_col = None
    
    # 0순위: Session State 확인 (Data Load 모드에서 방금 실행된 경우)
    if 'interpolation_results' in st.session_state and st.session_state.get('mm_data_ready', False):
        try:
            results = st.session_state['interpolation_results']
            df_fitted = results['interp_df'].copy()
            rfu_col = 'RFU_Interpolated' if 'RFU_Interpolated' in df_fitted.columns else 'RFU_Calculated'
            st.sidebar.success("✅ Data Load 모드 결과 적용됨 (메모리)")
            st.success("결과적용됨")
        except Exception as e:
            # 메모리 로드 실패 시 파일 로드 시도
            pass

    if uploaded_file is not None:
        # 업로드된 파일 처리
        import tempfile
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}', mode='wb') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == 'xlsx':
                # XLSX 파일: "Michaelis-Menten Curves" 시트 읽기
                df_fitted = pd.read_excel(tmp_path, sheet_name='Michaelis-Menten Curves', engine='openpyxl')
                rfu_col = 'RFU_Interpolated' if 'RFU_Interpolated' in df_fitted.columns else 'RFU_Calculated'
                st.sidebar.success("✅ 업로드된 XLSX 파일 사용 중 (Michaelis-Menten Curves 시트)")
            else:
                # CSV 파일
                df_fitted = pd.read_csv(tmp_path)
                rfu_col = 'RFU_Interpolated' if 'RFU_Interpolated' in df_fitted.columns else 'RFU_Calculated'
                st.sidebar.success("✅ 업로드된 CSV 파일 사용 중")
        finally:
            os.unlink(tmp_path)
    else:
        # Data Load 모드에서 생성된 결과 파일 자동 로드 (1순위: XLSX, 2순위: CSV)
        import os
        from pathlib import Path
        
        df_fitted = None
        
        # 1순위: XLSX 파일 (Michaelis-Menten_calibration_results.xlsx)
        xlsx_paths = [
            'Michaelis-Menten_calibration_results.xlsx',
            str(Path(__file__).parent.parent / 'Michaelis-Menten_calibration_results.xlsx'),
        ]
        
        for path in xlsx_paths:
            try:
                if os.path.exists(path):
                    df_fitted = pd.read_excel(path, sheet_name='Michaelis-Menten Curves', engine='openpyxl')
                    rfu_col = 'RFU_Interpolated' if 'RFU_Interpolated' in df_fitted.columns else 'RFU_Calculated'
                    st.sidebar.info(f"✅ Data Load 모드 결과 XLSX 자동 로드됨")
                    break
            except Exception:
                continue
        
        # 2순위: CSV 파일
        if df_fitted is None:
            csv_paths = [
                'data_interpolation_mode/results/MM_interpolated_curves.csv',
                str(Path(__file__).parent.parent / 'data_interpolation_mode' / 'results' / 'MM_interpolated_curves.csv'),
            ]
            
            for path in csv_paths:
                try:
                    if os.path.exists(path):
                        df_fitted = pd.read_csv(path)
                        rfu_col = 'RFU_Interpolated' if 'RFU_Interpolated' in df_fitted.columns else 'RFU_Calculated'
                        st.sidebar.info(f"✅ Data Load 모드 결과 CSV 자동 로드됨")
                        break
                except Exception:
                    continue
        
        if df_fitted is None:
            st.error("Data Load 모드 결과 파일을 찾을 수 없습니다. 먼저 'Data Load 모드'를 실행하여 결과를 다운로드하거나 CSV/XLSX 파일을 업로드해주세요.")
            st.stop()
        
        # rfu_col이 아직 설정되지 않았으면 설정
        if rfu_col is None:
            if 'RFU_Interpolated' in df_fitted.columns:
                rfu_col = 'RFU_Interpolated'
            elif 'RFU_Calculated' in df_fitted.columns:
                rfu_col = 'RFU_Calculated'
            else:
                rfu_col = 'RFU_Interpolated'  # 기본값
    
    # 엑셀 파일의 보간된 곡선 데이터 사용
    # Detect RFU column name
    rfu_col = None
    if 'RFU_Calculated' in df_fitted.columns:
        rfu_col = 'RFU_Calculated'
    elif 'RFU_Interpolated' in df_fitted.columns:
        rfu_col = 'RFU_Interpolated'
    else:
        st.error("RFU 데이터 컬럼을 찾을 수 없습니다. (RFU_Calculated 또는 RFU_Interpolated)")
        st.stop()
    
    # 엑셀 파일의 데이터 변환
    df_raw_converted = []
    unique_times = sorted(df_fitted['Time_min'].unique())
    
    # 농도 컬럼 이름 감지 (우선순위: ug/mL -> uM -> Concentration)
    conc_col_name = 'Concentration'
    if 'Concentration [ug/mL]' in df_fitted.columns:
        conc_col_name = 'Concentration [ug/mL]'
    elif 'Concentration [μM]' in df_fitted.columns:
        conc_col_name = 'Concentration [μM]'
    elif 'Concentration' in df_fitted.columns:
        conc_col_name = 'Concentration'
    
    for time in unique_times:
        time_data = df_fitted[df_fitted['Time_min'] == time]
        
        # Create row for each concentration
        for _, row in time_data.iterrows():
            conc_val = row.get(conc_col_name, 0)
            rfu = row[rfu_col]
            
            df_raw_converted.append({
                'time_min': time,
                'enzyme_ugml': conc_val,
                'conc_col_name': conc_col_name, # 원래 컬럼 이름 저장
                'FL_intensity': rfu,
                'SD': 0  # 보간된 곡선 데이터는 SD 없음
            })
    
    df_raw = pd.DataFrame(df_raw_converted)
    
    # 시간 범위 저장
    original_time_max = df_raw['time_min'].max()
    
    # 데이터 정보
    unique_times = sorted(df_raw['time_min'].unique())
    unique_concs = sorted(df_raw['enzyme_ugml'].unique())
    st.sidebar.success(f"✅ {len(unique_concs)}개 농도 조건, {len(unique_times)}개 시간 포인트 로드됨")
    
    # Store data source type for later use
    st.session_state['data_source_type'] = 'Fitted Curves (from Data Load mode)'
    st.session_state['original_time_max'] = original_time_max
    # 원본 fitted 데이터 저장 (Data Load 모드와 동일한 그래프를 그리기 위해)
    # df_fitted는 보간된 곡선 데이터이므로 원본 데이터 플롯에 사용
    if df_fitted is not None:
        st.session_state['df_fitted_original'] = df_fitted
        # rfu_col도 저장 (원본 데이터 플롯용)
        if rfu_col is not None:
            st.session_state['rfu_col'] = rfu_col
        else:
            # rfu_col이 없으면 기본값 사용
            st.session_state['rfu_col'] = 'RFU_Interpolated'
    
    # MM Results 시트에서 F0, Fmax 직접 읽기
    fitted_params = None
    xlsx_path_for_mm_results = None
    
    # 0순위: Session State 확인 (Data Load 모드에서 방금 실행된 경우)
    if 'interpolation_results' in st.session_state and st.session_state.get('mm_data_ready', False):
        try:
            results = st.session_state['interpolation_results']
            
            # Normalization results에서 가져오기 (가장 정확함)
            if 'normalization_results' in results:
                fitted_params = {}
                norm_results = results['normalization_results']
                for conc_name, data in norm_results.items():
                    conc_val = data['concentration']
                    fitted_params[float(conc_val)] = {
                        'F0': float(data['F0']),
                        'Fmax': float(data['Fmax'])
                    }
            # 없으면 dataframe에서 가져오기
            elif 'mm_results_df' in results:
                df_mm = results['mm_results_df']
                fitted_params = {}
                # 농도 컬럼 찾기
                conc_col = None
                for col in df_mm.columns:
                    if 'Concentration' in col:
                        conc_col = col
                        break
                
                if conc_col:
                    for _, row in df_mm.iterrows():
                        if pd.notna(row['F0']) and pd.notna(row['Fmax']):
                            fitted_params[float(row[conc_col])] = {
                                'F0': float(row['F0']),
                                'Fmax': float(row['Fmax'])
                            }
            
            if fitted_params and len(fitted_params) > 0:
                st.session_state['fitted_params'] = fitted_params
                st.sidebar.success(f"✅ 메모리에서 F0, Fmax 파라미터 적용됨 ({len(fitted_params)}개 농도)")
        except Exception as e:
            pass

    # 메모리 로드가 안되었거나 강제 업로드된 파일이 있는 경우 파일 처리 로직 수행
    if fitted_params is None:
        # 업로드된 파일 또는 자동 로드된 파일 경로 확인
        if uploaded_file is not None:
            import tempfile
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'xlsx':
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx', mode='wb') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    xlsx_path_for_mm_results = tmp_file.name
        else:
            # 자동 로드된 파일 경로 사용
            xlsx_paths = [
                'Michaelis-Menten_calibration_results.xlsx',
                str(Path(__file__).parent.parent / 'Michaelis-Menten_calibration_results.xlsx'),
            ]
            for path in xlsx_paths:
                if os.path.exists(path):
                    xlsx_path_for_mm_results = path
                    break
    
    # MM Results 시트 읽기
    if xlsx_path_for_mm_results is not None:
        try:
            # 1. MM Results (F0, Fmax, v0)
            df_mm_results = pd.read_excel(xlsx_path_for_mm_results, sheet_name='MM Results', engine='openpyxl')
            
            if df_mm_results is not None and 'F0' in df_mm_results.columns and 'Fmax' in df_mm_results.columns:
                fitted_params = {}
                conc_col_name = 'Concentration [ug/mL]' if 'Concentration [ug/mL]' in df_mm_results.columns else 'Concentration'
                
                # v0 data extraction lists
                v0_concs = []
                v0_vals = []
                
                for _, row in df_mm_results.iterrows():
                    conc_value = row[conc_col_name]
                    if pd.notna(conc_value):
                        try:
                            conc_float = float(conc_value)
                            # F0, Fmax
                            if pd.notna(row['F0']) and pd.notna(row['Fmax']):
                                fitted_params[conc_float] = {
                                    'F0': float(row['F0']),
                                    'Fmax': float(row['Fmax'])
                                }
                            
                            # v0
                            v0_val = row.get('v0') if 'v0' in row else row.get('v0 (RFU/min)')
                            if pd.notna(v0_val):
                                v0_concs.append(conc_float)
                                v0_vals.append(float(v0_val))
                        except (ValueError, TypeError):
                            continue
                
                if len(fitted_params) > 0:
                    st.sidebar.success(f"✅ {len(fitted_params)}개 농도 조건의 F0, Fmax 파라미터 로드 완료 (MM Results 시트)")
                    st.session_state['fitted_params'] = fitted_params
                else:
                    fitted_params = None
                    st.session_state['fitted_params'] = None
                
                # Store v0 data from file
                if v0_concs and v0_vals:
                    st.session_state['v0_data_from_file'] = {
                        'concentrations': v0_concs,
                        'v0_values': v0_vals
                    }
            else:
                fitted_params = None
                st.session_state['fitted_params'] = None

            # 2. MM Fit Results (Vmax, Km)
            try:
                xl = pd.ExcelFile(xlsx_path_for_mm_results)
                if 'MM Fit Results' in xl.sheet_names:
                    df_fit = pd.read_excel(xlsx_path_for_mm_results, sheet_name='MM Fit Results', engine='openpyxl')
                    mm_fit_from_file = {}
                    
                    # Determine columns
                    p_col = '파라미터' if '파라미터' in df_fit.columns else 'Parameter'
                    v_col = '값' if '값' in df_fit.columns else 'Value'
                    
                    if p_col in df_fit.columns and v_col in df_fit.columns:
                        params = dict(zip(df_fit[p_col], df_fit[v_col]))
                        
                        def get_param(keys):
                            for k in keys:
                                found = next((p for p in params if k.lower() in str(p).lower()), None)
                                if found: return params[found]
                            return None
                        
                        vmax = get_param(['Vmax'])
                        km = get_param(['Km'])
                        r2 = get_param(['R²', 'R2', 'R_squared'])
                        slope = get_param(['Slope'])
                        intercept = get_param(['Intercept'])
                        
                        def to_float(x):
                            try: return float(x)
                            except: return None
                            
                        if vmax is not None and km is not None:
                            mm_fit_from_file = {
                                'Vmax': to_float(vmax),
                                'Km': to_float(km),
                                'R_squared': to_float(r2),
                                'fit_success': True,
                                'experiment_type': "Substrate 농도 변화 (표준 MM)",
                                'equation': f"v₀ = {to_float(vmax):.2f}[S] / ({to_float(km):.2f} + [S])"
                            }
                        elif slope is not None:
                            mm_fit_from_file = {
                                'slope': to_float(slope),
                                'intercept': to_float(intercept) if intercept else 0,
                                'R_squared': to_float(r2),
                                'fit_success': True,
                                'experiment_type': "Enzyme 농도 변화",
                                'equation': f"v₀ = {to_float(slope):.4f}[E] + {to_float(intercept) if intercept else 0:.4f}"
                            }
                    
                    if mm_fit_from_file:
                         st.session_state['mm_fit_from_file'] = mm_fit_from_file
            except Exception:
                pass

        except Exception:
            fitted_params = None
            st.session_state['fitted_params'] = None
        finally:
            # 임시 파일 삭제
            if uploaded_file is not None and xlsx_path_for_mm_results and os.path.exists(xlsx_path_for_mm_results):
                try:
                    os.unlink(xlsx_path_for_mm_results)
                except:
                    pass
    else:
        fitted_params = None
        st.session_state['fitted_params'] = None
    
    # Step 2: Standardize units
    standardizer = UnitStandardizer(enzyme_mw=enzyme_mw)
    df_standardized = standardizer.standardize(df_raw)
    
    # Store time unit for later use
    time_unit = 'min' if 'time_min' in df_raw.columns else 's'
    st.session_state['time_unit'] = time_unit
    
    # Step 3-4: Normalization and region division
    normalizer = DataNormalizer()
    region_divider = RegionDivider()
    
    # Step 3-1: Initial temporary normalization (model-free threshold or fitted params)
    df_current = normalizer.normalize_temporary(df_standardized, fitted_params=fitted_params)
    
    # Step 4: Divide regions
    df_current = region_divider.divide_regions(df_current)
    
    # Step 3-2: Final normalization (using region information or fitted params)
    df_current = normalizer.normalize_final(df_current, fitted_params=fitted_params)
    
    df = df_current
    
    # Display data
    st.subheader("📊 데이터 미리보기")
    
    # Detect original column names for display
    time_unit = st.session_state.get('time_unit', 'min')
    # 원본 시간 범위 사용 (보간된 데이터가 아닌)
    original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
    if time_unit == 'min':
        time_display = f"0 - {original_time_max:.0f} 분"
        time_label = "시간 (분)"
    else:
        time_display = f"0 - {original_time_max:.0f} 초" if original_time_max < 100 else f"0 - {original_time_max/60:.1f} 분"
        time_label = "시간 (초)"
    # Determine concentration unit from normalized data
    # conc_col_name 컬럼이 있으면 사용, 없으면 enzyme_ugml 사용
    conc_col = 'enzyme_ugml'
    
    # 원래 컬럼 이름에 따른 단위 결정
    original_conc_col = df['conc_col_name'].iloc[0] if 'conc_col_name' in df.columns else 'Concentration [ug/mL]'
    
    if 'uM' in original_conc_col:
        conc_unit = "μM"
    elif 'nM' in original_conc_col:
        conc_unit = "nM"
    else:
        conc_unit = "μg/mL"
    
    st.session_state['time_label'] = time_label
    st.session_state['conc_unit'] = conc_unit
    
    col1, col2 = st.columns(2)
    with col1:
        # 중복 제거된 농도 조건 수 계산
        unique_concs = sorted(df[conc_col].unique())
        st.metric("농도 조건 수", len(unique_concs))
    with col2:
        st.metric("시간 범위", time_display)
    
    # Tabs for different views
    tab1, tab_alpha, tab2, tab_desc, tab3, tab4 = st.tabs([
        "📊 v₀ vs [S] Fit", 
        "📈 알파 계산",
        "🔬 모델 피팅",
        "📖 모델 설명",
        "📉 모델 비교",
        "💡 진단 분석"
    ])
    
    with tab1:
        # v0 vs [S] Michaelis-Menten Fit Graph
        st.subheader("v₀ vs [S] Michaelis-Menten Fit")
        
        # Data preparation
        v0_data = None
        mm_fit = None
        norm_results_data = None
        
        # 1. Try from session state (Memory from Data Load mode)
        if 'interpolation_results' in st.session_state and st.session_state.get('mm_data_ready', False):
            results = st.session_state['interpolation_results']
            if 'v0_vs_concentration' in results and 'mm_fit_results' in results:
                v0_data = results['v0_vs_concentration']
                mm_fit = results['mm_fit_results']
            
            # 정규화 결과 데이터 가져오기 (농도 값으로 변환)
            if 'normalization_results' in results:
                norm_results_raw = results['normalization_results']
                norm_results_data = {}
                for conc_name, data in norm_results_raw.items():
                    if 'concentration' in data:
                        conc_val = float(data['concentration'])
                        norm_results_data[conc_val] = {
                            'concentration': conc_val,
                            'F0': data.get('F0', None),
                            'Fmax': data.get('Fmax', None),
                            'k_obs': data.get('k_obs', None),
                            'tau': data.get('tau', None),
                            'R_squared': data.get('R_squared', None),
                            'equation': data.get('equation', None)
                        }
        
        # 2. Try from session state (Loaded from file in this mode)
        if (v0_data is None or mm_fit is None) and 'v0_data_from_file' in st.session_state:
            v0_data = st.session_state['v0_data_from_file']
            if 'mm_fit_from_file' in st.session_state:
                mm_fit = st.session_state['mm_fit_from_file']
        
        # 3. 파일에서 정규화 결과 읽기 (Normalization Results 시트 또는 MM Results 시트)
        if norm_results_data is None:
            xlsx_path_for_norm = None
            if uploaded_file is not None:
                import tempfile
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'xlsx':
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx', mode='wb') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        xlsx_path_for_norm = tmp_file.name
            else:
                xlsx_paths = [
                    'Michaelis-Menten_calibration_results.xlsx',
                    str(Path(__file__).parent.parent / 'Michaelis-Menten_calibration_results.xlsx'),
                ]
                for path in xlsx_paths:
                    if os.path.exists(path):
                        xlsx_path_for_norm = path
                        break
            
            if xlsx_path_for_norm is not None:
                try:
                    # Normalization Results 시트 시도
                    xl = pd.ExcelFile(xlsx_path_for_norm)
                    if 'Normalization Results' in xl.sheet_names:
                        df_norm = pd.read_excel(xlsx_path_for_norm, sheet_name='Normalization Results', engine='openpyxl')
                        norm_results_data = {}
                        for _, row in df_norm.iterrows():
                            # 농도 추출 (농도 컬럼 찾기)
                            conc_col = None
                            for col in df_norm.columns:
                                if '농도' in col or 'Concentration' in col:
                                    conc_col = col
                                    break
                            
                            if conc_col and pd.notna(row.get(conc_col)):
                                try:
                                    conc_val = float(str(row[conc_col]).replace('μM', '').replace('μg/mL', '').strip())
                                    norm_results_data[conc_val] = {
                                        'concentration': conc_val,
                                        'F0': row.get('F₀', row.get('F0', None)),
                                        'Fmax': row.get('F_max', row.get('Fmax', None)),
                                        'k_obs': row.get('k_obs', None),
                                        'tau': row.get('τ', row.get('tau', None)),
                                        'R_squared': row.get('R²', row.get('R_squared', None)),
                                        'equation': row.get('방정식', row.get('equation', None))
                                    }
                                except (ValueError, TypeError):
                                    continue
                    # Normalization Results 시트가 없으면 MM Results 시트에서 가져오기
                    elif 'MM Results' in xl.sheet_names:
                        df_mm = pd.read_excel(xlsx_path_for_norm, sheet_name='MM Results', engine='openpyxl')
                        norm_results_data = {}
                        conc_col_name = 'Concentration [ug/mL]' if 'Concentration [ug/mL]' in df_mm.columns else 'Concentration'
                        for _, row in df_mm.iterrows():
                            if pd.notna(row.get(conc_col_name)):
                                try:
                                    conc_val = float(row[conc_col_name])
                                    norm_results_data[conc_val] = {
                                        'concentration': conc_val,
                                        'F0': row.get('F0', None),
                                        'Fmax': row.get('Fmax', None),
                                        'k_obs': row.get('k_obs', None),
                                        'tau': row.get('tau', row.get('τ', None)),
                                        'R_squared': row.get('R²', row.get('R_squared', None)),
                                        'equation': row.get('방정식', row.get('equation', None))
                                    }
                                except (ValueError, TypeError):
                                    continue
                except Exception:
                    pass
                finally:
                    if uploaded_file is not None and xlsx_path_for_norm and os.path.exists(xlsx_path_for_norm):
                        try:
                            os.unlink(xlsx_path_for_norm)
                        except:
                            pass

        # Plotting
        if v0_data and mm_fit:
            # Determine exp type
            exp_type = mm_fit.get('experiment_type', 'Substrate 농도 변화 (표준 MM)')
            
            fig_v0 = go.Figure()
            
            # Experimental Points
            fig_v0.add_trace(go.Scatter(
                x=v0_data['concentrations'],
                y=v0_data['v0_values'],
                mode='markers',
                name='Experimental v₀',
                marker=dict(size=10, color='red', line=dict(width=2, color='black'))
            ))
            
            # Fit Line
            if mm_fit.get('fit_success'):
                conc_min = min(v0_data['concentrations'])
                conc_max = max(v0_data['concentrations'])
                conc_range = np.linspace(conc_min * 0.5, conc_max * 1.5, 200)
                
                if exp_type == "Substrate 농도 변화 (표준 MM)" and mm_fit.get('Vmax') is not None:
                     v0_fitted = michaelis_menten_calibration(conc_range, mm_fit['Vmax'], mm_fit['Km'])
                     line_name = mm_fit.get('equation', 'MM Fit')
                     
                     # Stats text
                     stats_text = f"Vmax = {mm_fit['Vmax']:.2f}<br>"
                     stats_text += f"Km = {mm_fit['Km']:.4f} μM<br>"
                     if mm_fit.get('R_squared'):
                        stats_text += f"R² = {mm_fit['R_squared']:.4f}"
                     
                     xaxis_title = '[S] (μM)'
                     title = 'Initial Velocity (v₀) vs Substrate Concentration [S]'
                     
                else: # Linear/Enzyme
                     slope = mm_fit.get('slope', 0)
                     intercept = mm_fit.get('intercept', 0)
                     v0_fitted = slope * conc_range + intercept
                     line_name = mm_fit.get('equation', 'Linear Fit')
                     
                     # Stats text
                     stats_text = f"Slope = {slope:.4f}<br>"
                     stats_text += f"Intercept = {intercept:.4f}<br>"
                     if mm_fit.get('R_squared'):
                        stats_text += f"R² = {mm_fit['R_squared']:.4f}<br>"
                     stats_text += "<br><b>⚠️ Km을 구할 수 없음</b>"
                     
                     xaxis_title = '[E] (μg/mL)'
                     title = 'Initial Velocity (v₀) vs Enzyme Concentration [E] (Substrate 고정)'

                fig_v0.add_trace(go.Scatter(
                    x=conc_range,
                    y=v0_fitted,
                    mode='lines',
                    name=line_name,
                    line=dict(width=2.5, color='blue')
                ))
                
                fig_v0.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    xanchor='left', yanchor='top',
                    text=stats_text,
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    font=dict(size=11)
                )

            fig_v0.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title='Initial Velocity v₀ (Fluorescence Units / Time)',
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_v0, use_container_width=True)
            
            # Show table with additional columns
            st.subheader("📋 실험 데이터")
            
            # 테이블 데이터 준비
            table_data = {
                xaxis_title: v0_data['concentrations'],
                'v₀ (RFU/min)': v0_data['v0_values']
            }
            
            # 정규화 결과 데이터 추가
            if norm_results_data:
                fmax_list = []
                r2_list = []
                k_obs_list = []
                tau_list = []
                equation_list = []
                
                for conc in v0_data['concentrations']:
                    # 농도 매칭 (부동소수점 오차 고려)
                    matched_data = None
                    for norm_conc, norm_data in norm_results_data.items():
                        if abs(float(conc) - float(norm_conc)) < 0.001:
                            matched_data = norm_data
                            break
                    
                    if matched_data:
                        fmax_list.append(matched_data.get('Fmax', None))
                        r2_list.append(matched_data.get('R_squared', None))
                        k_obs_list.append(matched_data.get('k_obs', None))
                        tau_list.append(matched_data.get('tau', None))
                        equation_list.append(matched_data.get('equation', None))
                    else:
                        fmax_list.append(None)
                        r2_list.append(None)
                        k_obs_list.append(None)
                        tau_list.append(None)
                        equation_list.append(None)
                
                table_data['Fmax'] = fmax_list
                table_data['R²'] = r2_list
                table_data['k_obs'] = k_obs_list
                table_data['τ'] = tau_list
                table_data['방정식'] = equation_list
            
            df_table = pd.DataFrame(table_data).sort_values(xaxis_title)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
                 
        else:
            st.info("⚠️ Michaelis-Menten 피팅 데이터가 없습니다. Data Load 모드에서 분석을 수행하거나 결과 파일(MM Fit Results 시트 포함)을 로드해주세요.")
    
    with tab_alpha:
        st.subheader("📈 알파(α) 계산")
        
        st.markdown("""
        **알파(α)란?**  
        정규화된 절단 비율로, 0 (절단 없음)에서 1 (완전 절단) 사이의 값을 가집니다.
        
        **계산식**: α(t) = (F(t) - F₀) / (Fmax - F₀)
        - **F(t)**: 시간 t에서의 형광값
          - Data Load 모드 결과 사용 시: 정규화를 통해 얻은 exponential 곡선의 interpolated 값 (RFU_Interpolated)
          - 직접 계산 시: 원본 데이터의 형광값
        - **F₀**: 초기 형광값
          - Data Load 모드에서 계산된 값이 있으면: 정규화 exponential 곡선의 interpolated 값들에서 얻은 F0 값 (MM Results 시트)
          - 없으면: 각 농도별 최소 형광값 (min(F))
        - **Fmax**: 최대 형광값
          - Data Load 모드에서 계산된 값이 있으면: 정규화 exponential 곡선의 interpolated 값들에서 얻은 Fmax 값 (MM Results 시트)
          - 없으면: Region-based 정규화 방식 사용
            1. Plateau 구간이 존재하면: Plateau 구간의 평균 형광값 (mean(F_plateau))
            2. 지수 증가 구간이 충분하면 (≥3점): 지수 함수 피팅으로 F∞ 계산 (F(t) = F₀ + A·(1 - e^(-k·t))에서 Fmax = F₀ + A)
            3. 그 외: 최대 형광값 (max(F))
        """)
        
        # Check if alpha column exists
        if 'alpha' not in df.columns:
            st.error("❌ Alpha 값이 계산되지 않았습니다. 데이터 정규화가 필요합니다.")
            st.info("💡 데이터가 정규화되지 않았습니다. 데이터 로드 및 정규화 과정을 확인해주세요.")
        else:
            # Alpha vs Time Plot
            st.subheader("📊 정규화 데이터: α(t) vs 시간")
            
            fig_alpha = Visualizer.plot_normalized_data(df, conc_unit, time_label, 
                                                       use_lines=True,
                                                       enzyme_name=enzyme_name,
                                                       substrate_name=substrate_name)
            # 원본 시간 범위로 xaxis 설정
            original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
            if time_unit == 'min':
                fig_alpha.update_xaxes(range=[0, original_time_max])
            else:
                fig_alpha.update_xaxes(range=[0, original_time_max])
            st.plotly_chart(fig_alpha, use_container_width=True)
            
            # Alpha Statistics
            st.subheader("📋 농도별 Alpha 통계")
            
            conc_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else df['conc_col_name'].iloc[0] if 'conc_col_name' in df.columns else None
            
            if conc_col:
                alpha_stats = []
                for conc in sorted(df[conc_col].unique()):
                    subset = df[df[conc_col] == conc]
                    alpha_stats.append({
                        f'농도 ({conc_unit})': conc,
                        'Alpha 최소값': f"{subset['alpha'].min():.4f}",
                        'Alpha 최대값': f"{subset['alpha'].max():.4f}",
                        'Alpha 평균': f"{subset['alpha'].mean():.4f}",
                        'Alpha 표준편차': f"{subset['alpha'].std():.4f}",
                        '데이터 포인트 수': len(subset)
                    })
                
                st.dataframe(pd.DataFrame(alpha_stats), use_container_width=True, hide_index=True)
            
            # F0, Fmax 정보
            st.subheader("🔬 정규화 파라미터 (F₀, Fmax)")
            
            # Check if fitted parameters are being used
            fitted_params_used = st.session_state.get('fitted_params', None)
            using_fitted_params = fitted_params_used is not None and len(fitted_params_used) > 0
            
            if using_fitted_params:
                st.success(f"✅ F0, Fmax 파라미터 로드 완료 ({len(fitted_params_used)}개 농도 조건)")
                st.info("💡 F0, Fmax 값은 Data Load 모드의 정규화를 통해 얻은 exponential 곡선의 interpolated 값들에서 계산된 값입니다 (MM Results 시트).")
            else:
                st.info("ℹ️ 기본 정규화 방식 사용 중 (interpolated 값에서 F0, Fmax 계산)")
            
            # F0, Fmax 테이블
            if conc_col and 'F0' in df.columns and 'Fmax' in df.columns:
                f0_fmax_data = []
                for conc in sorted(df[conc_col].unique()):
                    subset = df[df[conc_col] == conc]
                    fmax_method = subset['Fmax_method'].iloc[0] if 'Fmax_method' in subset.columns else "N/A"
                    
                    f0_fmax_data.append({
                        f'농도 ({conc_unit})': conc,
                        'F₀ (초기)': f"{subset['F0'].iloc[0]:.2f}",
                        'Fmax (최대)': f"{subset['Fmax'].iloc[0]:.2f}",
                        'Fmax 방법': fmax_method,
                        'Alpha 범위': f"{subset['alpha'].min():.3f} - {subset['alpha'].max():.3f}"
                    })
                
                st.dataframe(pd.DataFrame(f0_fmax_data), use_container_width=True, hide_index=True)
            
            # 정규화 방법 설명
            with st.expander("📖 정규화 방법 상세 설명", expanded=False):
                if using_fitted_params:
                    st.markdown("""
                    **MM Results 시트에서 F0, Fmax 사용:**
                    - F0, Fmax: Data Load 모드에서 생성된 MM Results 시트에서 직접 읽어옴
                    - 곡선: F(t) = F₀ + (Fmax - F₀)·[1 - exp(-k·t)]
                    - α(t) = (F(t) − F₀) / (Fmax − F₀)
                    - Data Load 모드에서 이미 계산된 파라미터를 그대로 사용
                    """)
                else:
                    st.markdown("""
                    **기본 정규화 방식 (Region-based):**
                    
                    1. **임시 정규화 (Temporary Normalization)**
                       - F0_temp = 최소 형광값 (min(F))
                       - Fmax_temp = 최대 형광값 (max(F))
                       - α_temp = (F - F0_temp) / (Fmax_temp - F0_temp)
                    
                    2. **구간 구분 (Region Division)**
                       - 초기 선형 구간 (Initial Linear Region)
                       - 지수 증가 구간 (Exponential Growth Region)
                       - Plateau 구간 (Plateau Region)
                    
                    3. **최종 정규화 (Final Normalization)**
                       - F0 = F0_temp (최소값 유지)
                       - Fmax 결정 방법:
                         * Plateau 구간이 있으면 → Plateau 평균값
                         * 지수 증가 구간이 충분하면 → 지수 피팅으로 F∞ 계산
                         * 그 외 → 최대값 사용
                       - α = (F - F0) / (Fmax - F0)
                    
                    **Fmax 방법 설명:**
                    - `plateau_avg`: Plateau 구간의 평균값 사용
                    - `exponential_fit`: 지수 함수 피팅으로 계산된 F∞ 사용
                    - `fallback_max`: 최대값 사용 (fallback)
                    """)
            
            # Alpha 데이터 다운로드
            st.subheader("💾 Alpha 데이터 다운로드")
            
            # Alpha 데이터 준비
            alpha_download_df = df[['time_s', conc_col, 'alpha', 'F0', 'Fmax']].copy() if conc_col else df[['time_s', 'alpha', 'F0', 'Fmax']].copy()
            alpha_download_df = alpha_download_df.sort_values(['time_s', conc_col] if conc_col else 'time_s')
            
            csv_alpha = alpha_download_df.to_csv(index=False)
            st.download_button(
                label="📥 Alpha 데이터 다운로드 (CSV)",
                data=csv_alpha,
                file_name="alpha_calculation_results.csv",
                mime="text/csv",
                help="시간, 농도, alpha, F0, Fmax 값을 포함한 CSV 파일"
            )
    
    with tab2:
        st.subheader("🔬 글로벌 모델 피팅")
        
        st.markdown("""
        **기본 모델 (A-C)**: 고전적 효소 키네틱 메커니즘  
        **확장 모델 (D-F)**: Fmax 농도 의존성 설명 (겔 침투, 생성물 억제, 효소 흡착)
        """)
        
        # Model selection
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**기본 모델**")
            fit_model_a = st.checkbox("모델 A: 기질 고갈", value=True)
            st.caption("✓ 1차 반응 및 기질 고갈")
            
            fit_model_b = st.checkbox("모델 B: 효소 비활성화", value=True)
            st.caption("✓ 효소 비활성화 & 시간 의존")
            
            fit_model_c = st.checkbox("모델 C: 물질전달 제한", value=True)
            st.caption("✓ 확산 제한 & 접근성 제약")
        
        with col2:
            st.markdown("**확장 모델 (Fmax 의존성)**")
            fit_model_d = st.checkbox("모델 D: 농도 의존 Fmax", value=True)
            st.caption("✓ 겔 침투 깊이 & 2차 절단")
            
            fit_model_e = st.checkbox("모델 E: 생성물 억제", value=True)
            st.caption("✓ 생성물 축적 & 경쟁 억제")
            
            fit_model_f = st.checkbox("모델 F: 효소 흡착/격리", value=True)
            st.caption("✓ 표면 흡착 & 비가역 결합")
        
        if st.button("🚀 글로벌 피팅 실행", type="primary"):
            # 데이터 상태 확인 및 검증
            with st.expander("🔍 데이터 상태 확인", expanded=False):
                st.write("**필수 컬럼 확인:**")
                required_cols = ['alpha', 'time_s', 'FL_intensity']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ 누락된 컬럼: {missing_cols}")
                    st.stop()
                else:
                    st.success(f"✅ 필수 컬럼 존재: {required_cols}")
                
                st.write("**데이터 통계:**")
                st.write(f"- 전체 데이터 포인트: {len(df)}")
                st.write(f"- Alpha 범위: {df['alpha'].min():.4f} ~ {df['alpha'].max():.4f}")
                st.write(f"- Alpha 평균: {df['alpha'].mean():.4f}")
                st.write(f"- Alpha 표준편차: {df['alpha'].std():.4f}")
                st.write(f"- 시간 범위: {df['time_s'].min():.2f} ~ {df['time_s'].max():.2f} 초")
                
                # 농도별 alpha 분포 확인
                conc_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else df['conc_col_name'].iloc[0] if 'conc_col_name' in df.columns else None
                if conc_col:
                    st.write(f"**농도별 Alpha 통계:**")
                    conc_stats = df.groupby(conc_col)['alpha'].agg(['count', 'min', 'max', 'mean', 'std'])
                    st.dataframe(conc_stats, use_container_width=True)
                
                # 문제가 있는 데이터 확인
                if df['alpha'].max() < 0.1:
                    st.warning("⚠️ Alpha 값이 모두 0.1 미만입니다. 정규화가 제대로 되지 않았을 수 있습니다.")
                if df['alpha'].std() < 0.01:
                    st.warning("⚠️ Alpha 값의 변동성이 매우 작습니다. 데이터가 제대로 정규화되지 않았을 수 있습니다.")
            
            results = []
            
            # Create a status container
            status_container = st.empty()
            result_container = st.container()
            
            # Model A
            if fit_model_a:
                with status_container:
                    with st.spinner("🔄 모델 A: 기질 고갈 피팅 중..."):
                        model_a = ModelA_SubstrateDepletion(enzyme_mw=enzyme_mw)
                        result_a = model_a.fit_global(df, verbose_callback=verbose_callback)
                        results.append(result_a)
                
                if result_a:
                    with result_container:
                        st.success(f"✅ 모델 A 완료: R² = {result_a.r_squared:.4f}, AIC = {result_a.aic:.2f}")
                else:
                    with result_container:
                        st.error("❌ 모델 A 피팅 실패")
            
            # Model B
            if fit_model_b:
                with status_container:
                    with st.spinner("🔄 모델 B: 효소 비활성화 피팅 중..."):
                        model_b = ModelB_EnzymeDeactivation(enzyme_mw=enzyme_mw)
                        result_b = model_b.fit_global(df, verbose_callback=verbose_callback)
                        results.append(result_b)
                
                if result_b:
                    with result_container:
                        st.success(f"✅ 모델 B 완료: R² = {result_b.r_squared:.4f}, AIC = {result_b.aic:.2f}")
                else:
                    with result_container:
                        st.error("❌ 모델 B 피팅 실패")
            
            # Model C
            if fit_model_c:
                with status_container:
                    with st.spinner("🔄 모델 C: 물질전달 제한 피팅 중..."):
                        model_c = ModelC_MassTransfer(enzyme_mw=enzyme_mw)
                        result_c = model_c.fit_global(df, verbose_callback=verbose_callback)
                        results.append(result_c)
                
                if result_c:
                    with result_container:
                        st.success(f"✅ 모델 C 완료: R² = {result_c.r_squared:.4f}, AIC = {result_c.aic:.2f}")
                else:
                    with result_container:
                        st.error("❌ 모델 C 피팅 실패")
            
            # Model D
            if fit_model_d:
                with status_container:
                    with st.spinner("🔄 모델 D: 농도 의존 Fmax 피팅 중..."):
                        model_d = ModelD_ConcentrationDependentFmax(enzyme_mw=enzyme_mw)
                        result_d = model_d.fit_global(df, verbose_callback=verbose_callback)
                        results.append(result_d)
                
                if result_d:
                    with result_container:
                        st.success(f"✅ 모델 D 완료: R² = {result_d.r_squared:.4f}, AIC = {result_d.aic:.2f}")
                else:
                    with result_container:
                        st.error("❌ 모델 D 피팅 실패")
            
            # Model E
            if fit_model_e:
                with status_container:
                    with st.spinner("🔄 모델 E: 생성물 억제 피팅 중..."):
                        model_e = ModelE_ProductInhibition(enzyme_mw=enzyme_mw)
                        result_e = model_e.fit_global(df, verbose_callback=verbose_callback)
                        results.append(result_e)
                
                if result_e:
                    with result_container:
                        st.success(f"✅ 모델 E 완료: R² = {result_e.r_squared:.4f}, AIC = {result_e.aic:.2f}")
                else:
                    with result_container:
                        st.error("❌ 모델 E 피팅 실패")
            
            # Model F
            if fit_model_f:
                with status_container:
                    with st.spinner("🔄 모델 F: 효소 흡착/격리 피팅 중..."):
                        model_f = ModelF_EnzymeSurfaceSequestration(enzyme_mw=enzyme_mw)
                        result_f = model_f.fit_global(df, verbose_callback=verbose_callback)
                        results.append(result_f)
                
                if result_f:
                    with result_container:
                        st.success(f"✅ 모델 F 완료: R² = {result_f.r_squared:.4f}, AIC = {result_f.aic:.2f}")
                else:
                    with result_container:
                        st.error("❌ 모델 F 피팅 실패")
            
            # Clear status container after all done
            status_container.empty()
            
            # Store results in session state
            st.session_state['fit_results'] = results
            st.session_state['df'] = df
            
            # Show completion message
            with result_container:
                st.success("🎉 모든 모델 피팅 완료! '모델 비교' 탭에서 결과를 확인하세요.")
    
    with tab_desc:
        st.subheader("📚 키네틱 모델 상세 설명")
        st.markdown(r"""
        이 시뮬레이터는 펩타이드 기질과 효소 반응을 분석하기 위해 6가지 키네틱 모델을 제공합니다.
        
        #### 1. 기본 모델 (Basic Models)
        고전적인 효소 반응 속도론을 기반으로 하며, Fmax가 효소 농도에 독립적인 경우를 가정합니다.

        **📌 모델 A: 기질 고갈 (Substrate Depletion)**
        - **개요**: 가장 기본적인 1차 반응 모델입니다. 기질([S])이 소모됨에 따라 반응 속도가 감소합니다.
        - **수식**:
          $$ \alpha(t) = 1 - e^{-k_{obs} \cdot t} $$
          $$ k_{obs} = \frac{k_{cat}}{K_M} \cdot [E] $$
        - **파라미터**:
          - $k_{cat}/K_M$: 촉매 효율 ($M^{-1}s^{-1}$)
        - **특징**: 
          - [E]가 낮을 때 초기 속도 v₀는 [E]에 선형 비례합니다.
          - 시간이 지나면 모든 기질이 절단되어 정규화된 형광값 α가 1에 도달합니다.

        **📌 모델 B: 효소 비활성화 (Enzyme Deactivation)**
        - **개요**: 반응 진행 중 효소가 서서히 활성을 잃는 현상을 설명합니다.
        - **수식**:
          효소 농도가 지수적으로 감소한다고 가정 ($[E]_t = [E]_0 \cdot e^{-k_d t}$)
          $$ \alpha(t) = 1 - \exp\left[-\frac{k_{cat}/K_M \cdot [E]_0}{k_d} (1 - e^{-k_d t})\right] $$
        - **파라미터**:
          - $k_{cat}/K_M$: 촉매 효율 ($M^{-1}s^{-1}$)
          - $k_d$: 효소 비활성화 속도 상수 ($s^{-1}$)
        - **특징**:
          - 반응 곡선이 예상보다 일찍 평형에 도달하며, 기질이 남아있음에도 반응이 멈출 수 있습니다 ($\alpha_{\infty} < 1$).

        **📌 모델 C: 물질전달 제한 (Mass Transfer Limitation)**
        - **개요**: 효소가 기질 표면으로 확산되는 속도가 반응 속도보다 느린 경우입니다.
        - **수식**:
          표면 효소 농도 $[E]_s$는 벌크 농도 $[E]_b$보다 낮음
          $$ [E]_s \approx \frac{[E]_b}{1 + Da}, \quad Da = \frac{k_{cat} \Gamma_0}{K_M k_m} $$
          $$ \alpha(t) = 1 - e^{-k_{obs} \cdot t}, \quad k_{obs} = \frac{k_{cat}}{K_M} [E]_s $$
        - **파라미터**:
          - $k_{cat}/K_M$: 촉매 효율
          - $k_m$: 물질전달 계수 ($m/s$)
          - $\Gamma_0$: 초기 표면 기질 밀도 ($pmol/cm^2$)
        - **특징**:
          - 초기 반응 속도가 확산에 의해 제한되므로, 효소 농도가 높아져도 반응 속도가 비례해서 증가하지 않고 포화됩니다.

        ---

        #### 2. 확장 모델 (Extended Models)
        Fmax(최대 형광값)가 효소 농도([E])에 따라 달라지는 복잡한 표면 반응을 설명합니다.

        **📌 모델 D: 농도 의존 Fmax (Concentration Dependent Fmax)**
        - **개요**: 효소 농도가 높을수록 더 많은 기질에 접근할 수 있는 경우(침투 깊이 증가 등)입니다.
        - **수식**:
          최대 절단율 $\alpha_{max}$가 효소 농도에 의존
          $$ \alpha(t) = \alpha_{max}([E]) \cdot (1 - e^{-k_{obs} t}) $$
          $$ \alpha_{max}([E]) = \alpha_{\infty} \cdot (1 - e^{-k_{access} [E]}) $$
        - **파라미터**:
          - $k_{cat}/K_M$: 촉매 효율
          - $\alpha_{\infty}$: 이론적 최대 절단 비율
          - $k_{access}$: 접근성 계수 ($M^{-1}$)
        - **특징**:
          - 낮은 [E]에서는 표면 기질만 절단되지만, 높은 [E]에서는 내부 기질까지 절단되어 최종 형광값(Fmax)이 증가합니다.

        **📌 모델 E: 생성물 억제 (Product Inhibition)**
        - **개요**: 반응 생성물이 효소의 활성 부위에 결합하여 반응을 방해하는 경우입니다.
        - **수식**:
          경쟁적 억제 모델을 미분방정식으로 풀이
          $$ \frac{d\alpha}{dt} = \frac{k_{obs}(1-\alpha)}{1 + K_{i,eff}\alpha} $$
        - **파라미터**:
          - $k_{cat}/K_M$: 촉매 효율
          - $K_{i,eff}$: 유효 억제 상수 (무차원, $[S]_0/K_i$)
        - **특징**:
          - 반응 초기에 비해 후반부 속도가 급격히 감소합니다.

        **📌 모델 F: 효소 흡착/격리 (Enzyme Surface Sequestration)**
        - **개요**: 효소가 기질 표면이나 겔 매트릭스에 비가역적으로 흡착되어 반응에 참여하지 못하는 경우입니다.
        - **수식**:
          효소가 표면에 흡착되어($k_{ads}$) 고갈됨
          $$ \alpha(t) \approx \frac{(k_{cat}/K_M)[E]}{k_{ads}(1+K_{ads}[E])} (1 - e^{-k_{ads} t}) $$
        - **파라미터**:
          - $k_{cat}/K_M$: 촉매 효율
          - $k_{ads}$: 흡착 속도 상수 ($s^{-1}$)
          - $K_{ads}$: 흡착 평형 상수 ($M^{-1}$)
        - **특징**:
          - 높은 [E]에서도 예상보다 낮은 반응성을 보일 수 있습니다.
          
        ### 📊 모델 적합도 평가 기준 (AIC)

        **Akaike Information Criterion (AIC)**  
        모델의 적합도(Goodness of fit)와 복잡도(Complexity) 사이의 균형을 평가하는 지표입니다. 값이 작을수록 더 좋은 모델로 간주합니다.

        **계산식**:
        $$ AIC = 2k - 2\ln(\hat{L}) $$
        여기서:
        - $k$: 모델의 파라미터 수
        - $\hat{L}$: 모델의 최대 우도(Maximum Likelihood)

        본 프로그램에서는 잔차 제곱합(RSS)을 이용하여 다음과 같이 계산합니다:
        $$ AIC = n \ln\left(\frac{RSS}{n}\right) + 2k + C $$
        - $n$: 전체 데이터 포인트 수
        - $RSS$: 잔차 제곱합 ($\sum (y_{obs} - y_{pred})^2$)
        - $C$: 상수항 (전체 우도 식 포함)

        **해석**:
        - **ΔAIC < 2**: 두 모델 간 유의미한 차이가 없음
        - **ΔAIC > 10**: AIC가 낮은 모델이 통계적으로 훨씬 더 적합함
        """)

    with tab3:
        if 'fit_results' in st.session_state:
            results = st.session_state['fit_results']
            df = st.session_state['df']
            
            st.subheader("📊 모델 비교")
            
            # Comparison table
            comparison_df = Visualizer.create_comparison_table(results)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Determine best model
            valid_results = [r for r in results if r is not None]
            if valid_results:
                best_aic = min(r.aic for r in valid_results)
                best_model = [r for r in valid_results if r.aic == best_aic][0]
                
                st.success(f"🏆 최적 모델 (최저 AIC): **{best_model.name}** (AIC = {best_model.aic:.2f})")
                
                # Parameter details for best model
                st.subheader(f"최적 모델 파라미터: {best_model.name}")
                param_data = []
                for param, value in best_model.params.items():
                    std = best_model.params_std.get(param, 0)
                    param_data.append({
                        '파라미터': param,
                        '값': f"{value:.4e}",
                        '표준오차': f"{std:.4e}",
                        '상대오차': f"{(std/value*100):.2f}%" if value != 0 else "N/A"
                    })
                st.dataframe(pd.DataFrame(param_data), use_container_width=True)
            
            # Plot all model fits
            st.subheader("📈 전체 모델 피팅 결과")
            fig_models = Visualizer.plot_model_fits(df, results, conc_unit, time_label,
                                                    enzyme_name=enzyme_name,
                                                    substrate_name=substrate_name)
            # 원본 시간 범위로 xaxis 설정
            original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
            if time_unit == 'min':
                fig_models.update_xaxes(range=[0, original_time_max])
            else:
                fig_models.update_xaxes(range=[0, original_time_max])
            st.plotly_chart(fig_models, use_container_width=True)
            
            # Individual model plots
            st.subheader("📊 개별 모델 비교")
            st.markdown("각 모델별로 원본 데이터와 피팅 결과를 비교합니다.")
            
            # Create tabs for each model
            model_names = [r.name for r in results if r is not None]
            
            if len(model_names) > 0:
                model_tabs_ui = st.tabs(model_names)
                
                for idx, (tab, result) in enumerate(zip(model_tabs_ui, [r for r in results if r is not None])):
                    with tab:
                        # Color scheme for each model
                        model_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
                        color = model_colors[idx % len(model_colors)]
                        
                        # Display individual model plot
                        fig_ind = Visualizer.plot_individual_model(df, result, conc_unit, time_label, color)
                        # 원본 시간 범위로 xaxis 설정
                        original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
                        if time_unit == 'min':
                            fig_ind.update_xaxes(range=[0, original_time_max])
                        else:
                            fig_ind.update_xaxes(range=[0, original_time_max])
                        st.plotly_chart(fig_ind, use_container_width=True)
                        
                        # Display parameters
                        st.markdown(f"**{result.name} 파라미터**")
                        param_cols = st.columns(len(result.params))
                        for col_idx, (param, value) in enumerate(result.params.items()):
                            with param_cols[col_idx]:
                                std = result.params_std.get(param, 0)
                                st.metric(
                                    label=param,
                                    value=f"{value:.4e}",
                                    delta=f"±{std:.4e}" if std > 0 else None
                                )
            
            # Download results
            st.subheader("💾 결과 다운로드")
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="비교 테이블 다운로드 (CSV)",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )
        else:
            st.info("👈 먼저 '모델 피팅' 탭에서 피팅을 실행해주세요.")
    
    with tab4:
        st.subheader("💡 진단 분석")
        
        # Initial rate analysis
        st.plotly_chart(
            Visualizer.plot_initial_rates(df, conc_unit, time_unit), 
            use_container_width=True
        )
        
        st.markdown("""
        ### 📋 모델 선택 가이드라인
        
        #### 기본 모델 (A-C)
        
        **모델 A (기질 고갈)** 선호 조건:
        - 초기 속도 v₀가 [E]에 대해 선형 관계 (낮은 [E]에서)
        - 포화 형광 F∞ ≈ 일정 (정규화된 α → 1)
        - 유의미한 효소 비활성화가 관찰되지 않음
        
        **모델 B (효소 비활성화)** 선호 조건:
        - F∞ < 이론적 최대값 (포화에서 α < 1)
        - 빠른 초기 증가 후 예상보다 낮은 수준에서 평탄화
        - kd > 0이며 유의미한 기여도
        
        **모델 C (물질전달 제한)** 선호 조건:
        - 초기 버스트(0-5초) 후 느린 접근
        - 교반/유속에 민감
        - 높은 [E]에서 v₀ vs [E] 그래프가 포화 양상
        
        #### 확장 모델 (D-F): **Fmax가 [E]에 따라 변하는 경우**
        
        **모델 D (농도 의존 Fmax)** 선호 조건:
        - 높은 [E]에서 α_max 증가 (더 많은 기질 접근)
        - 겔 침투 깊이 효과 (두꺼운/밀집 겔)
        - 2차 절단으로 생성물 방출 증가
        - **파라미터**: α_∞ (최대값), k_access (접근성 계수)
        
        **모델 E (생성물 억제)** 선호 조건:
        - 초기 빠른 증가 후 감속 (생성물 축적)
        - 낮은 [E]에서 더 큰 억제 효과
        - 생성물 제거 시 반응 속도 회복
        - **파라미터**: Ki_eff (억제 상수)
        
        **모델 F (효소 흡착/격리)** 선호 조건:
        - 높은 [E]에서 상대적으로 덜 영향받음 (포화)
        - 음전하 표면/PDA 코팅, 밀집 겔 구조
        - 시간에 따른 효소 활성 감소 (비가역)
        - **파라미터**: k_ads (흡착속도), K_ads (평형상수)
        
        ### 📊 통계 기준
        - **AIC/BIC**: 낮을수록 좋음 (파라미터 수 페널티)
        - **R²**: 높을수록 좋음 (>0.95 우수)
        - **RMSE**: 낮을수록 좋음
        - **Δ AIC > 10**: 높은 AIC 모델에 대한 강력한 반증
        - **Δ AIC < 2**: 모델 간 유의미한 차이 없음
        """)
        
        # Experimental suggestions
        st.subheader("🧪 제안 후속 실험 (모델 구분)")
        
        st.markdown("""
        ### 🔍 Fmax 농도 의존성 확인 실험
        
        1. **다양한 [E]에서 장시간 측정** (30분-1시간)
           - 각 농도별 포화 형광값(Fmax) 정량 측정
           - [E] vs Fmax 플롯 → 선형/포화 양상 확인
           - **선형 증가** → 모델 D 가능성
           - **일정** → 기본 모델 A-C
        
        2. **겔 두께 변화 테스트** (모델 D)
           - 얇은 겔(50 μm) vs 두꺼운 겔(500 μm)
           - 두꺼운 겔에서 [E] 의존성 증가 → 확산 침투 제한
           - 얇은 겔에서 [E] 독립적 → 표면 반응 우세
        
        3. **생성물 첨가 실험** (모델 E)
           - 미리 절단된 펩타이드 조각 첨가
           - 반응 초기 속도 감소 → 생성물 억제 증명
           - 높은 [생성물]에서 α_max 감소 관찰
        
        4. **표면 처리 변화** (모델 F)
           - 양전하 표면 vs 음전하(PDA) vs 중성(PEG)
           - 음전하 표면에서 [E] 의존성 강화 → 흡착 증명
           - PEG 표면에서 흡착 감소 → 모델 D/E로 전환
        
        ### 🧬 고전적 메커니즘 테스트
        
        5. **Pulse-chase 실험** (모델 B)
           - t=5분에 신선한 효소 재투입
           - 곡선 재상승 → 기질 남음 (모델 A)
           - 변화 없음 → 효소 비활성화 (모델 B)
        
        6. **교반/유속 변화** (모델 C)
           - 정적 vs 회전 (100 rpm) vs 관류 (1 mL/min)
           - 유속 증가로 α 증가 → 물질전달 제한
           - 변화 없음 → 반응속도 제한 (모델 A/B)
        
        7. **기질 밀도 변화** (모델 A)
           - 0.5배, 1배, 2배 펩타이드 고정화
           - Fmax 비례 증가 → 기질 고갈
           - Fmax 불변 → 다른 메커니즘 우세
        
        8. **용액상 대조실험**
           - 가용성 기질 (같은 농도)
           - 완전 절단(α→1) → 표면/확산 문제
           - 불완전 절단 → 본질적 억제/비활성화
        
        ### 🎯 모델 결정 트리
        
        ```
        Fmax가 [E]에 따라 증가하는가?
        ├─ YES → 확장 모델 (D-F) 테스트
        │   ├─ 겔 두께 민감? → 모델 D (침투)
        │   ├─ 생성물 첨가로 감소? → 모델 E (억제)
        │   └─ 표면 전하 민감? → 모델 F (흡착)
        │
        └─ NO → 기본 모델 (A-C) 테스트
            ├─ Pulse-chase 반응? → 모델 A (기질)
            ├─ 시간에 따라 α_max↓? → 모델 B (비활성)
            └─ 유속에 민감? → 모델 C (확산)
        ```
        """)


