#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw data만 입력받아 Michaelis-Menten Fitting 후 calibration curve 생성
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정 (GUI 없이 PNG 저장)
import warnings
warnings.filterwarnings('ignore')


def read_raw_data(filename='mode_prep_raw_data/raw.csv'):
    """
    raw.csv/xlsx에서 원본 데이터 읽기 및 정리
    
    지원하는 형식:
    1. 기존 형식 (탭 구분):
       - 첫 번째 행: 농도 값들 (각 농도가 mean, SD, N으로 3번 반복)
       - 두 번째 행: 컬럼 헤더 (time_min, mean, SD, N, mean, SD, N, ...)
       - 세 번째 행부터: 실제 데이터
    
    2. 새로운 형식 (쉼표 구분):
       - 첫 번째 행: 헤더 (concentration_uM, min, RFU_min, SD, N)
       - 데이터: 각 행이 농도, 시간, RFU, SD, N
    """
    # 파일 확장자 확인
    file_extension = filename.split('.')[-1].lower()
    
    # CSV 또는 XLSX 파일 읽기
    if file_extension == 'xlsx':
        # XLSX 파일 읽기
        # 첫 번째 행만 읽어서 농도 값 추출
        first_row_df = pd.read_excel(filename, header=None, nrows=1, engine='openpyxl')
        concentration_row = first_row_df.iloc[0].values[1:]  # 첫 번째 컬럼(빈 값) 제외
        
        # 두 번째 행을 헤더로 읽기
        header_row_df = pd.read_excel(filename, header=None, skiprows=[0], nrows=1, engine='openpyxl')
        header_names = header_row_df.iloc[0].values
        
        # 세 번째 행부터 데이터로 읽기 (헤더 없이)
        df = pd.read_excel(filename, header=None, skiprows=[0, 1], engine='openpyxl')
        
        # 헤더 이름 설정
        df.columns = header_names
    else:
        # CSV 파일 읽기 - 형식 자동 감지
        # 먼저 쉼표 구분자로 시도 (새 형식)
        try:
            df_test = pd.read_csv(filename, nrows=1)
            # 새 형식 감지: concentration_uM, min, RFU_min 등의 컬럼이 있는지 확인
            if 'concentration_uM' in df_test.columns or 'concentration' in df_test.columns:
                # 새 형식: 첫 번째 행이 헤더
                df = pd.read_csv(filename)
                # 새 형식 처리
                return _read_new_format_csv(df)
        except:
            pass
        
        # 기존 형식 시도 (탭 구분자)
        try:
            # 첫 번째 행만 읽어서 농도 값 추출
            first_row_df = pd.read_csv(filename, header=None, nrows=1, sep='\t')
            concentration_row = first_row_df.iloc[0].values[1:]  # 첫 번째 컬럼(빈 값) 제외
            
            # 두 번째 행을 헤더로 읽기
            header_row_df = pd.read_csv(filename, header=None, skiprows=[0], nrows=1, sep='\t')
            header_names = header_row_df.iloc[0].values
            
            # 세 번째 행부터 데이터로 읽기 (헤더 없이)
            df = pd.read_csv(filename, header=None, skiprows=[0, 1], sep='\t')
        except Exception as e:
            # 탭 구분자 실패 시 쉼표 구분자로 재시도
            df = pd.read_csv(filename)
            # 새 형식 처리
            return _read_new_format_csv(df)
        
        # 헤더 이름 설정
        df.columns = header_names
    
    # 첫 번째 컬럼이 시간
    time_col = df.columns[0]
    times = pd.to_numeric(df[time_col].values, errors='coerce')
    
    # 농도별 데이터 추출
    data = {}
    i = 1  # 첫 번째 데이터 컬럼부터 시작
    conc_idx = 0  # 농도 인덱스
    
    while i < len(df.columns):
        # 농도 값은 첫 번째 행에서 가져옴 (mean, SD, N 중 mean 위치)
        if conc_idx < len(concentration_row):
            conc_value = float(concentration_row[conc_idx * 3])  # 각 농도의 첫 번째 값 (mean 위치)
        else:
            conc_value = float(concentration_row[conc_idx * 3]) if len(concentration_row) > conc_idx * 3 else conc_idx
        
        # 컬럼명 생성
        conc_name = f"{conc_value} ug/mL"
        
        # mean 컬럼 (값)
        value_col_idx = i
        # SD 컬럼
        sd_col_idx = i + 1 if i + 1 < len(df.columns) else None
        # N 컬럼 (사용 안 함, 건너뜀)
        
        value_col = df.columns[value_col_idx]
        sd_col = df.columns[sd_col_idx] if sd_col_idx is not None else None
        
        # NaN이 아닌 값만 추출 (0 값도 포함)
        # 컬럼 인덱스로 직접 접근하여 Series 추출
        value_series = df.iloc[:, value_col_idx]
        valid_mask = (~pd.isna(value_series)) & (value_series.astype(str) != '') & (value_series.astype(str) != 'nan')
        valid_mask = valid_mask.values.flatten() if hasattr(valid_mask, 'values') else np.array(valid_mask).flatten()
        
        # 유효한 행만 필터링 (인덱스로 직접 접근)
        valid_indices = np.where(valid_mask)[0]
        valid_times = pd.to_numeric(df.iloc[valid_indices, 0], errors='coerce').values
        valid_values = pd.to_numeric(df.iloc[valid_indices, value_col_idx], errors='coerce').values
        if sd_col_idx is not None:
            valid_sd = pd.to_numeric(df.iloc[valid_indices, sd_col_idx], errors='coerce').values
        else:
            valid_sd = None
        
        # NaN만 제거 (0 값은 유지)
        valid_mask2 = ~pd.isna(valid_values)
        valid_mask2 = np.array(valid_mask2)  # numpy array로 변환
        valid_times = valid_times[valid_mask2]
        valid_values = valid_values[valid_mask2]
        if valid_sd is not None:
            valid_sd = valid_sd[valid_mask2]
        
        if len(valid_times) > 0:
            data[conc_name] = {
                'time': valid_times,
                'value': valid_values,
                'SD': valid_sd,
                'concentration': conc_value,
                'conc_name': conc_name
            }
        
        # 다음 농도로 (3개 컬럼씩: mean, SD, N)
        i += 3
        conc_idx += 1
    
    return data


def _read_new_format_csv(df):
    """
    새로운 형식의 CSV 파일 읽기
    형식: concentration_uM, min, RFU_min, SD, N
    """
    data = {}
    
    # 컬럼명 확인 및 정규화
    conc_col = None
    time_col = None
    rfu_col = None
    sd_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        # 더 구체적인 매칭 (순서 중요)
        if col_lower == 'min' or col_lower == 'time' or col_lower == 'time_min':
            time_col = col
        elif 'rfu' in col_lower or 'fluorescence' in col_lower or ('fl' in col_lower and 'intensity' in col_lower):
            rfu_col = col
        elif 'concentration' in col_lower or 'conc' in col_lower:
            conc_col = col
        elif col_lower == 'sd' or col_lower == 'std' or 'standard' in col_lower:
            sd_col = col
    
    # 컬럼명이 정확히 일치하는 경우 우선 처리
    if 'min' in df.columns:
        time_col = 'min'
    if 'concentration_uM' in df.columns:
        conc_col = 'concentration_uM'
    if 'RFU_min' in df.columns:
        rfu_col = 'RFU_min'
    if 'SD' in df.columns:
        sd_col = 'SD'
    
    if conc_col is None or time_col is None or rfu_col is None:
        raise ValueError(f"필수 컬럼을 찾을 수 없습니다. 발견된 컬럼: {df.columns.tolist()}")
    
    # 농도별로 그룹화
    for conc_value in df[conc_col].unique():
        if pd.isna(conc_value):
            continue
        
        conc_subset = df[df[conc_col] == conc_value].copy()
        
        # 시간과 RFU 값 추출
        times = pd.to_numeric(conc_subset[time_col].values, errors='coerce')
        values = pd.to_numeric(conc_subset[rfu_col].values, errors='coerce')
        
        # SD 값 추출 (있는 경우)
        if sd_col and sd_col in conc_subset.columns:
            sd_values = pd.to_numeric(conc_subset[sd_col].values, errors='coerce')
        else:
            sd_values = None
        
        # NaN 제거
        valid_mask = ~pd.isna(times) & ~pd.isna(values)
        valid_times = times[valid_mask]
        valid_values = values[valid_mask]
        if sd_values is not None:
            valid_sd = sd_values[valid_mask]
        else:
            valid_sd = None
        
        if len(valid_times) > 0:
            # 농도 값 정규화 (uM 단위로 통일)
            try:
                conc_float = float(conc_value)
            except:
                conc_float = float(conc_value)
            
            # 컬럼명에 uM이 있으면 μM 단위로, 없으면 ug/mL 단위로
            if 'um' in conc_col.lower() or 'uM' in conc_col or 'μM' in conc_col:
                conc_name = f"{conc_float} μM"
            else:
                conc_name = f"{conc_float} ug/mL"
            
            data[conc_name] = {
                'time': valid_times,
                'value': valid_values,
                'SD': valid_sd,
                'concentration': conc_float,
                'conc_name': conc_name
            }
    
    return data


def calculate_initial_velocity(times, values, linear_fraction=0.2, min_points=3):
    """
    Quenched peptide protease kinetics: 초기 속도 계산
    
    시간-형광 그래프에서 선형 구간의 기울기를 계산하여 초기 속도(v0)를 구합니다.
    
    Parameters:
    - times: 시간 배열 (분 또는 초)
    - values: 형광값 배열
    - linear_fraction: 선형 구간으로 사용할 초기 데이터 비율 (기본값: 0.2 = 처음 20%)
    - min_points: 최소 데이터 포인트 수
    
    Returns:
    - v0: 초기 속도 (형광 단위/시간 단위)
    - F0: 초기 형광값 (y절편)
    - r_squared: 선형 피팅의 R²
    - linear_times: 선형 구간 시간 배열
    - linear_values: 선형 구간 형광값 배열
    """
    times = np.array(times)
    values = np.array(values)
    
    # 정렬 (시간 순서대로)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    
    # 선형 구간 결정: 초기 데이터의 linear_fraction만큼 사용
    n_total = len(times)
    n_linear = max(min_points, int(n_total * linear_fraction))
    
    # 최소한 min_points 이상이어야 함
    if n_linear < min_points or n_total < min_points:
        # 데이터가 부족하면 가능한 만큼 사용
        n_linear = min(min_points, n_total)
    
    linear_times = times[:n_linear]
    linear_values = values[:n_linear]
    
    # 선형 피팅: F(t) = F0 + v0 * t
    if len(linear_times) >= 2 and np.ptp(linear_times) > 0:
        coeffs = np.polyfit(linear_times, linear_values, 1)
        v0 = coeffs[0]  # 기울기 = 초기 속도
        F0 = coeffs[1]  # y절편 = 초기 형광값
        
        # R² 계산
        fit_values = np.polyval(coeffs, linear_times)
        ss_res = np.sum((linear_values - fit_values) ** 2)
        ss_tot = np.sum((linear_values - np.mean(linear_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        # 데이터 부족 시 단순 계산
        if len(linear_times) >= 2:
            v0 = (linear_values[-1] - linear_values[0]) / (linear_times[-1] - linear_times[0])
        else:
            v0 = 0
        F0 = values[0] if len(values) > 0 else 0
        r_squared = 0
    
    return v0, F0, r_squared, linear_times, linear_values


def fit_time_course(times, values, model='linear'):
    """
    Quenched peptide protease kinetics: 초기 속도 계산
    
    시간-형광 그래프에서 선형 구간의 기울기를 계산합니다.
    MM 방정식은 시간-형광 데이터에 직접 적용하지 않습니다.
    
    Parameters:
    - times: 시간 배열
    - values: 형광값 배열
    - model: 'linear' (선형 구간 분석만 수행)
    
    Returns:
    - params: 피팅 파라미터 딕셔너리 (v0, F0 포함)
    - fit_values: 선형 피팅된 값 (선형 구간만)
    - r_squared: 선형 피팅의 R²
    """
    # 초기 속도 계산 (선형 구간 분석)
    v0, F0, r_squared, linear_times, linear_values = calculate_initial_velocity(times, values)
    
    # 선형 피팅 값 생성 (선형 구간에 대해서만)
    if len(linear_times) >= 2:
        coeffs = np.polyfit(linear_times, linear_values, 1)
        fit_values_linear = np.polyval(coeffs, linear_times)
    else:
        fit_values_linear = linear_values
    
    # 전체 시간에 대한 피팅 값 (선형 구간만 표시용)
    fit_values = np.full_like(times, np.nan)
    fit_values[:len(linear_times)] = fit_values_linear
    
    # Fmax는 전체 데이터의 최대값
    Fmax = np.max(values) if len(values) > 0 else F0
    
    params = {
        'v0': v0,  # 초기 속도 (형광 단위/시간 단위)
        'F0': F0,  # 초기 형광값
        'Fmax': Fmax,  # 최대 형광값
        'R_squared': r_squared,  # 선형 피팅의 R²
        'linear_fraction': len(linear_times) / len(times) if len(times) > 0 else 0
    }
    
    return params, fit_values, r_squared


def michaelis_menten_calibration(x, Vmax_cal, Km_cal):
    """
    Calibration Curve: Michaelis-Menten 방정식
    y = (Vmax * x) / (Km + x)
    
    Parameters:
    - x: 농도
    - Vmax_cal: 최대 응답
    - Km_cal: 반속도 농도 (Michaelis 상수)
    """
    return (Vmax_cal * x) / (Km_cal + x)


def fit_calibration_curve(concentrations, responses):
    """
    농도 vs 응답 데이터에 calibration curve 피팅
    
    Parameters:
    - concentrations: 농도 배열
    - responses: 응답 배열 (Vmax 또는 형광값)
    
    Returns:
    - params: Vmax_cal, Km_cal
    - fit_values: 피팅된 값
    - equation: 방정식 문자열
    """
    concentrations = np.array(concentrations)
    responses = np.array(responses)
    
    # 초기값 추정
    Vmax_init = np.max(responses)
    Km_init = np.mean(concentrations)
    
    try:
        popt, pcov = curve_fit(
            michaelis_menten_calibration,
            concentrations, responses,
            p0=[Vmax_init, Km_init],
            bounds=([0, 0.01], [np.inf, np.inf]),
            maxfev=5000
        )
        
        Vmax_cal, Km_cal = popt
        perr = np.sqrt(np.diag(pcov))
        
        fit_values = michaelis_menten_calibration(concentrations, Vmax_cal, Km_cal)
        
        # R² 계산
        ss_res = np.sum((responses - fit_values) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        equation = f"y = ({Vmax_cal:.2f} * x) / ({Km_cal:.4f} + x)"
        
        params = {
            'Vmax_cal': Vmax_cal,
            'Km_cal': Km_cal,
            'Vmax_cal_std': perr[0],
            'Km_cal_std': perr[1],
            'R_squared': r_squared
        }
        
        return params, fit_values, equation
        
    except Exception as e:
        print(f"   ⚠️ Calibration curve 피팅 실패: {e}")
        # 선형 근사
        coeffs = np.polyfit(concentrations, responses, 1)
        fit_values = np.polyval(coeffs, concentrations)
        equation = f"y = {coeffs[0]:.2f} * x + {coeffs[1]:.2f}"
        
        params = {
            'Vmax_cal': coeffs[0],
            'Km_cal': 0,
            'R_squared': 0
        }
        
        return params, fit_values, equation


def main():
    """메인 함수"""
    print("📊 Michaelis-Menten Calibration Curve Generator")
    print("=" * 70)
    
    # 1. Raw data 읽기
    print("\n1️⃣ Raw data 파일 읽는 중...")
    try:
        raw_data = read_raw_data('mode_prep_raw_data/raw.csv')
        print(f"   ✅ {len(raw_data)}개 농도 조건 발견")
        for conc_name, data in raw_data.items():
            print(f"      - {conc_name}: {len(data['time'])}개 데이터 포인트")
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return
    
    # 2. 각 농도별 초기 속도 계산 (Quenched peptide protease kinetics)
    print("\n2️⃣ 각 농도별 초기 속도(v0) 계산 중...")
    print("   (시간-형광 그래프의 선형 구간에서 기울기 계산)")
    
    v0_results = {}
    all_fit_data = []
    
    for conc_name, data in raw_data.items():
        times = data['time']
        values = data['value']
        
        # 초기 속도 계산 (선형 구간 분석)
        params, fit_values, r_sq = fit_time_course(times, values, model='linear')
        
        # 초기 속도 파라미터 추출
        v0 = params['v0']  # 초기 속도 (형광 단위/시간 단위)
        F0 = params['F0']  # 초기 형광값
        Fmax = params['Fmax']  # 최대 형광값
        
        v0_results[conc_name] = {
            'concentration': data['concentration'],
            'v0': v0,  # 초기 속도
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
        
        print(f"   ✅ {conc_name}: v0={v0:.2f} (형광/시간), F0={F0:.2f}, Fmax={Fmax:.2f}, R²={r_sq:.4f}")
    
    # 3. 초기 속도 결과 CSV 저장
    print("\n3️⃣ 초기 속도 결과 CSV 생성 중...")
    
    results_data = []
    for conc_name, params in sorted(v0_results.items(), key=lambda x: x[1]['concentration']):
        results_data.append({
            'Concentration [ug/mL]': params['concentration'],
            'v0': params['v0'],  # 초기 속도
            'F0': params['F0'],
            'Fmax': params['Fmax'],
            'R_squared': params['R_squared'],
            'linear_fraction': params['linear_fraction']
        })
    
    results_df = pd.DataFrame(results_data)
    results_filename = 'prep_data/fitting_results/initial_velocity_results.csv'
    
    results_df.to_csv('prep_data/fitting_results/initial_velocity_detailed.csv', index=False)
    print(f"   ✅ prep_data/fitting_results/initial_velocity_detailed.csv 저장 완료 (상세 데이터)")
    
    # 4. Michaelis-Menten Calibration Curve 생성 (v0 vs [S])
    print("\n4️⃣ Michaelis-Menten Calibration Curve 생성 중...")
    print("   (초기 속도 v0 vs 농도 [S]에 MM 방정식 피팅)")
    
    # 농도 vs 초기 속도(v0)로 calibration curve 피팅
    concentrations = [v0_results[cn]['concentration'] for cn in sorted(v0_results.keys(), 
                                                                      key=lambda x: v0_results[x]['concentration'])]
    v0_values = [v0_results[cn]['v0'] for cn in sorted(v0_results.keys(), 
                                                      key=lambda x: v0_results[x]['concentration'])]
    
    # MM calibration curve 피팅: v0 = Vmax * [S] / (Km + [S])
    cal_params, cal_fit_values, cal_equation = fit_calibration_curve(concentrations, v0_values)
    
    print(f"   ✅ Calibration Equation: {cal_equation}")
    print(f"      Vmax = {cal_params['Vmax_cal']:.2f} ± {cal_params.get('Vmax_cal_std', 0):.2f} (형광 단위/시간 단위)")
    print(f"      Km = {cal_params['Km_cal']:.4f} ± {cal_params.get('Km_cal_std', 0):.4f} (μg/mL)")
    print(f"      R² = {cal_params['R_squared']:.4f}")
    
    # 5. Calibration Curve XY 데이터 생성
    print("\n5️⃣ Calibration Curve XY 데이터 생성 중...")
    
    # 고밀도 농도 범위
    conc_min = min(concentrations)
    conc_max = max(concentrations)
    conc_range = np.linspace(conc_min * 0.5, conc_max * 1.5, 200)
    
    # Calibration curve 계산: v0 = Vmax * [S] / (Km + [S])
    cal_y_values = michaelis_menten_calibration(conc_range, 
                                                cal_params['Vmax_cal'], 
                                                cal_params['Km_cal'])
    
    # Calibration curve 데이터 저장
    cal_curve_data = []
    for x, y in zip(conc_range, cal_y_values):
        cal_curve_data.append({
            'Concentration_ug/mL': x,
            'v0_Fitted': y,  # 초기 속도
            'Equation': cal_equation
        })
    
    cal_curve_df = pd.DataFrame(cal_curve_data)
    cal_curve_filename = 'prep_data/fitting_results/MM_calibration_curve.csv'
    cal_curve_df.to_csv(cal_curve_filename, index=False)
    print(f"   ✅ {cal_curve_filename} 저장 완료 ({len(cal_curve_df)} 행)")
    
    # 6. 선형 피팅 곡선 데이터 저장
    fit_curves_df = pd.DataFrame(all_fit_data)
    fit_curves_filename = 'prep_data/fitting_results/linear_fit_curves.csv'
    fit_curves_df.to_csv(fit_curves_filename, index=False)
    print(f"   ✅ {fit_curves_filename} 저장 완료 ({len(fit_curves_df)} 행)")
    
    # 7. 방정식 요약 저장
    print("\n6️⃣ 방정식 요약 저장 중...")
    
    equations_data = [{
        'Type': 'Calibration Curve (v0 vs [S])',
        'Equation': cal_equation,
        'Vmax': cal_params['Vmax_cal'],
        'Km': cal_params['Km_cal'],
        'R_squared': cal_params['R_squared']
    }]
    
    # 각 농도별 초기 속도 정보
    for conc_name, params in sorted(v0_results.items(), key=lambda x: x[1]['concentration']):
        eq = f"v0 = {params['v0']:.2f} (선형 구간 기울기)"
        equations_data.append({
            'Type': f'{conc_name}',
            'Equation': eq,
            'v0': params['v0'],
            'F0': params['F0'],
            'Fmax': params['Fmax'],
            'R_squared': params['R_squared']
        })
    
    equations_df = pd.DataFrame(equations_data)
    equations_filename = 'prep_data/fitting_results/MM_equations.csv'
    equations_df.to_csv(equations_filename, index=False)
    print(f"   ✅ {equations_filename} 저장 완료")
    
    # MM_calibration_equations.csv 형식으로 저장
    calibration_equations_data = []
    for conc_name, params in sorted(v0_results.items(), key=lambda x: x[1]['concentration']):
        calibration_equations_data.append({
            'Concentration': conc_name,
            'Concentration_ug/mL': params['concentration'],
            'v0': params['v0'],  # 초기 속도
            'F0': params['F0'],
            'Fmax': params['Fmax'],
            'R_squared': params['R_squared'],
            'linear_fraction': params['linear_fraction']
        })
    
    calibration_equations_df = pd.DataFrame(calibration_equations_data)
    calibration_equations_filename = 'prep_data/fitting_results/MM_calibration_equations.csv'
    calibration_equations_df.to_csv(calibration_equations_filename, index=False)
    print(f"   ✅ {calibration_equations_filename} 저장 완료 (농도별 초기 속도 데이터)")
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("📋 생성된 파일:")
    print(f"   1. prep_data/fitting_results/initial_velocity_detailed.csv - 초기 속도 상세 데이터")
    print(f"   2. {cal_curve_filename} - Calibration curve XY 데이터 (그래프용)")
    print(f"   3. prep_data/fitting_results/MM_calibration_curve.png - Calibration curve 그래프 (PNG)")
    print(f"   4. {fit_curves_filename} - 각 농도별 선형 피팅 데이터")
    print(f"   5. {equations_filename} - 모든 방정식 요약")
    print(f"   6. {calibration_equations_filename} - 농도별 초기 속도 데이터")
    print("\n📊 Michaelis-Menten Calibration Curve (v0 vs [S]):")
    print(f"   {cal_equation}")
    print(f"   농도 범위: {conc_min:.4f} - {conc_max:.4f} (확장: {conc_min*0.5:.4f} - {conc_max*1.5:.4f})")
    # 7. Calibration Curve 그래프 생성
    print("\n7️⃣ Calibration Curve 그래프 생성 중...")
    plot_calibration_curve(
        cal_curve_df, results_df, cal_params, cal_equation
    )
    print("   ✅ prep_data/fitting_results/MM_calibration_curve.png 저장 완료")
    
    print("\n✨ 완료!")


def plot_calibration_curve(cal_curve_df, results_df, cal_params, cal_equation):
    """
    Calibration curve 그래프를 그리고 PNG로 저장
    (v0 vs [S]에 MM 방정식 피팅)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calibration curve 그리기
    ax.plot(
        cal_curve_df['Concentration_ug/mL'],
        cal_curve_df['v0_Fitted'],
        'b-', linewidth=2.5,
        label=f'MM Fit: {cal_equation}',
        zorder=1
    )
    
    # 실험 데이터 포인트 그리기 (v0 vs [S])
    concentrations = results_df['Concentration [ug/mL]'].values
    v0_values = results_df['v0'].values
    
    ax.scatter(
        concentrations,
        v0_values,
        color='red',
        s=150,
        marker='o',
        edgecolors='black',
        linewidths=2,
        label='Experimental Data (v₀)',
        zorder=2
    )
    
    # 그래프 스타일
    ax.set_xlabel('Concentration [S] (μg/mL)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Initial Velocity v₀ (Fluorescence Units / Time)', fontsize=14, fontweight='bold')
    ax.set_title('Michaelis-Menten Calibration Curve\n(Initial Velocity v₀ vs Substrate Concentration)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 그리드 추가
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    # 통계 정보 텍스트 박스
    stats_text = f"Vmax = {cal_params['Vmax_cal']:.2f}\n"
    stats_text += f"Km = {cal_params['Km_cal']:.4f} μg/mL\n"
    stats_text += f"R² = {cal_params['R_squared']:.4f}"
    
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # PNG 저장
    plt.savefig('prep_data/fitting_results/MM_calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
