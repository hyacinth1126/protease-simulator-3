import pandas as pd
import streamlit as st

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
    """일반 분석 모드 - 표준 FRET 분석"""
    
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
    
    # 엑셀 파일의 데이터를 변환
    df_raw_converted = []
    unique_times = sorted(df_fitted['Time_min'].unique())
    
    for time in unique_times:
        time_data = df_fitted[df_fitted['Time_min'] == time]
        
        # Create row for each concentration
        for _, row in time_data.iterrows():
            conc_ugml = row.get('Concentration [ug/mL]', 0)
            rfu = row[rfu_col]
            
            df_raw_converted.append({
                'time_min': time,
                'enzyme_ugml': conc_ugml,
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
            df_mm_results = pd.read_excel(xlsx_path_for_mm_results, sheet_name='MM Results', engine='openpyxl')
            
            if df_mm_results is not None and 'F0' in df_mm_results.columns and 'Fmax' in df_mm_results.columns:
                fitted_params = {}
                conc_col_name = 'Concentration [ug/mL]' if 'Concentration [ug/mL]' in df_mm_results.columns else 'Concentration'
                
                for _, row in df_mm_results.iterrows():
                    conc_value = row[conc_col_name]
                    if pd.notna(conc_value) and pd.notna(row['F0']) and pd.notna(row['Fmax']):
                        try:
                            conc_float = float(conc_value)
                            fitted_params[conc_float] = {
                                'F0': float(row['F0']),
                                'Fmax': float(row['Fmax'])
                            }
                        except (ValueError, TypeError):
                            continue
                
                if len(fitted_params) > 0:
                    st.sidebar.success(f"✅ {len(fitted_params)}개 농도 조건의 F0, Fmax 파라미터 로드 완료 (MM Results 시트)")
                    st.session_state['fitted_params'] = fitted_params
                else:
                    fitted_params = None
                    st.session_state['fitted_params'] = None
            else:
                fitted_params = None
                st.session_state['fitted_params'] = None
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
    conc_col = df['conc_col_name'].iloc[0] if 'conc_col_name' in df.columns else 'enzyme_ugml'
    if 'uM' in conc_col:
        conc_unit = "μM"
    elif 'nM' in conc_col:
        conc_unit = "nM"
    else:
        conc_unit = "μg/mL"
    
    st.session_state['time_label'] = time_label
    st.session_state['conc_unit'] = conc_unit
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"농도 조건 ({conc_unit})", df[conc_col].nunique())
    with col2:
        st.metric("시간 범위", time_display)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 원본 데이터", 
        "📊 정규화 데이터", 
        "🔬 모델 피팅",
        "📉 모델 비교",
        "💡 진단 분석"
    ])
    
    with tab1:
        # Data Load 모드와 동일한 그래프를 그리기 위해 원본 fitted 데이터 사용
        if 'df_fitted_original' in st.session_state:
            df_fitted_orig = st.session_state['df_fitted_original']
            rfu_col = st.session_state.get('rfu_col', 'RFU_Interpolated')
            
            # Data Load 모드와 동일한 형식으로 그래프 생성
            import plotly.graph_objects as go
            fig_raw = go.Figure()
            colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # 농도 순서대로 정렬
            if 'Concentration [ug/mL]' in df_fitted_orig.columns:
                conc_order = df_fitted_orig.sort_values('Concentration [ug/mL]')['Concentration'].unique()
            else:
                conc_order = df_fitted_orig['Concentration'].unique()
            
            for idx, conc_name in enumerate(conc_order):
                color = colors[idx % len(colors)]
                subset = df_fitted_orig[df_fitted_orig['Concentration'] == conc_name]
                
                if len(subset) > 0:
                    fig_raw.add_trace(go.Scatter(
                        x=subset['Time_min'],
                        y=subset[rfu_col],
                        mode='lines',
                        name=conc_name,
                        line=dict(color=color, width=2.5),
                        legendgroup=conc_name,
                        showlegend=True
                    ))
            
            fig_raw.update_layout(
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
            
            # 원본 시간 범위로 xaxis 설정
            original_time_max = st.session_state.get('original_time_max', df_fitted_orig['Time_min'].max())
            fig_raw.update_xaxes(range=[0, original_time_max])
            fig_raw.update_yaxes(rangemode='tozero')
            
            st.plotly_chart(fig_raw, use_container_width=True)
        else:
            # 기존 방식 (fallback)
            fig_raw = Visualizer.plot_raw_data(df, conc_unit, time_label, 
                                              use_lines=True,
                                              enzyme_name=enzyme_name, 
                                              substrate_name=substrate_name)
            # 원본 시간 범위로 xaxis 설정
            original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
            if time_unit == 'min':
                fig_raw.update_xaxes(range=[0, original_time_max])
            else:
                fig_raw.update_xaxes(range=[0, original_time_max])
            st.plotly_chart(fig_raw, use_container_width=True)
        
        st.subheader("Raw data table")
        st.dataframe(df, height=400, use_container_width=True)
    
    with tab2:
        # Controls and method description for normalization
        st.subheader("정규화 설정 및 방법")
        
        # Check if fitted parameters are being used
        fitted_params_used = st.session_state.get('fitted_params', None)
        using_fitted_params = fitted_params_used is not None and len(fitted_params_used) > 0
        if using_fitted_params:
            st.success(f"✅ F0, Fmax 파라미터 로드 완료 ({len(fitted_params_used)}개 농도 조건)")
            st.info("💡 F0, Fmax 값은 MM Results 시트에서 가져온 값입니다.")
        else:
            st.info("ℹ️ 기본 정규화 방식 사용 중 (원본 데이터에서 F0, Fmax 계산)")
        
        with st.expander("정규화 방법 보기", expanded=False):
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
                **기본 정규화 방식:**
                - 각 농도별 지수 피팅: F(t) = F₀ + A·(1−e⁻ᵏᵗ)
                - 점근선 Fmax = F₀ + A 사용
                - α(t) = (F(t) − F₀) / (Fmax − F₀)
                """)

        fig_norm = Visualizer.plot_normalized_data(df, conc_unit, time_label, 
                                                   use_lines=True,
                                                   enzyme_name=enzyme_name,
                                                   substrate_name=substrate_name)
        # 원본 시간 범위로 xaxis 설정
        original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
        if time_unit == 'min':
            fig_norm.update_xaxes(range=[0, original_time_max])
        else:
            fig_norm.update_xaxes(range=[0, original_time_max])
        st.plotly_chart(fig_norm, use_container_width=True)
        
        # 시간-농도 그래프 추가
        st.subheader("시간-농도 그래프")
        fig_heatmap = Visualizer.plot_time_concentration_heatmap(df, conc_unit, time_label,
                                                                 enzyme_name=enzyme_name,
                                                                 substrate_name=substrate_name)
        # 원본 시간 범위로 xaxis 설정
        original_time_max = st.session_state.get('original_time_max', df['time_s'].max())
        if time_unit == 'min':
            fig_heatmap.update_xaxes(range=[0, original_time_max])
        else:
            fig_heatmap.update_xaxes(range=[0, original_time_max])
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Summary statistics
        fitted_params_used = st.session_state.get('fitted_params', None)
        if fitted_params_used is not None and len(fitted_params_used) > 0:
            st.subheader("정규화 요약 (MM Results 시트 사용)")
        else:
            st.subheader("정규화 요약 (지수 피팅 기반)")
        
        summary_data = []
        for conc in sorted(df[conc_col].unique()):
            subset = df[df[conc_col] == conc]
            # Check if optional columns exist
            fmax_std = f"{subset['Fmax_std'].iloc[0]:.1f}" if 'Fmax_std' in subset.columns else "N/A"
            fit_k = f"{subset['fit_k'].iloc[0]:.4f}" if 'fit_k' in subset.columns else "N/A"
            fmax_method = subset['Fmax_method'].iloc[0] if 'Fmax_method' in subset.columns else "N/A"
            
            summary_data.append({
                f'농도 ({conc_unit})': conc,
                'F0 (초기)': f"{subset['F0'].iloc[0]:.1f}",
                'Fmax (점근선)': f"{subset['Fmax'].iloc[0]:.1f}",
                'Fmax 방법': fmax_method,
                'Fmax 표준편차': fmax_std,
                '피팅 k (s⁻¹)': fit_k,
                'α 범위': f"{subset['alpha'].min():.3f} - {subset['alpha'].max():.3f}",
                'α 평균': f"{subset['alpha'].mean():.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        if fitted_params_used is not None and len(fitted_params_used) > 0:
            st.info("📊 F0, Fmax 값은 MM Results 시트에서 가져온 값입니다.")
        else:
            st.info("📊 각 농도별로 F(t) = F0 + A·(1-exp(-k·t)) 형태의 지수 함수를 피팅하여 점근선 Fmax를 결정합니다.")
    
    with tab3:
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
    
    with tab4:
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
    
    with tab5:
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


