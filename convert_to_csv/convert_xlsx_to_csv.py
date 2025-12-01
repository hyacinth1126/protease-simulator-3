#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XLSX 파일을 prep_raw.csv 형식으로 변환하는 스크립트
"""

import pandas as pd
import os
import sys
from pathlib import Path


def extract_concentration_from_header(header_value):
    """
    헤더에서 농도 값 추출
    예: "0.3125ug/ml" -> 0.3125
    """
    if pd.isna(header_value):
        return None
    
    header_str = str(header_value).strip()
    
    # 숫자 부분만 추출
    import re
    match = re.search(r'(\d+\.?\d*)', header_str)
    if match:
        return float(match.group(1))
    
    return None


def convert_xlsx_to_prep_raw(xlsx_path, output_path=None, n_value=50):
    """
    XLSX 파일을 prep_raw.csv 형식으로 변환
    
    Parameters:
    - xlsx_path: 입력 XLSX 파일 경로
    - output_path: 출력 CSV 파일 경로 (None이면 자동 생성)
    - n_value: N 값 (복제수, 기본값 50)
    
    Returns:
    - output_path: 생성된 CSV 파일 경로
    """
    # XLSX 파일 읽기
    print(f"📂 XLSX 파일 읽는 중: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name='Sheet1', engine='openpyxl', header=None)
    
    print(f"   파일 크기: {df.shape[0]}행 x {df.shape[1]}열")
    
    # 첫 번째 행: 컬럼 헤더 (min, 농도들)
    first_row = df.iloc[0].values
    time_col_name = first_row[0]  # 'min' 또는 'time_min'
    
    # 두 번째 행: 헤더 타입 (RFU, SD, N)
    second_row = df.iloc[1].values
    
    # 농도 값 추출 (각 농도마다 3개 컬럼: RFU, SD, N)
    concentrations = []
    for i in range(1, len(first_row), 3):  # RFU, SD, N 3개씩 구성
        if i < len(first_row):
            conc_value = extract_concentration_from_header(first_row[i])
            if conc_value is not None:
                concentrations.append(conc_value)
    
    print(f"   발견된 농도: {concentrations}")
    
    # 세 번째 행부터: 실제 데이터
    data_df = df.iloc[2:].copy()
    data_df.columns = first_row  # 첫 번째 행을 컬럼명으로 설정
    
    # 시간 컬럼 찾기
    time_col = data_df.columns[0]
    times = pd.to_numeric(data_df[time_col].values, errors='coerce')
    
    # prep_raw.csv 형식으로 변환
    output_lines = []
    
    # 첫 번째 행: 농도 값들 (각 농도가 mean, SD, N으로 3번 반복)
    first_line = ['']  # 첫 번째 컬럼은 빈 값
    for conc in concentrations:
        first_line.extend([str(conc), str(conc), str(conc)])
    output_lines.append('\t'.join(first_line))
    
    # 두 번째 행: 컬럼 헤더
    second_line = ['time_min']
    for conc in concentrations:
        second_line.extend(['mean', 'SD', 'N'])
    output_lines.append('\t'.join(second_line))
    
    # 세 번째 행부터: 실제 데이터
    for idx, time_val in enumerate(times):
        if pd.isna(time_val):
            continue
        
        data_line = [str(time_val)]
        
        # 각 농도별로 데이터 추출 (RFU, SD, N 3개 컬럼씩)
        for i, conc in enumerate(concentrations):
            # 각 농도마다 3개 컬럼: RFU, SD, N
            # 컬럼 인덱스: 1+3*i (RFU), 2+3*i (SD), 3+3*i (N)
            rfu_col_idx = 1 + i * 3
            sd_col_idx = 2 + i * 3
            n_col_idx = 3 + i * 3
            
            if rfu_col_idx < len(data_df.columns) and sd_col_idx < len(data_df.columns) and n_col_idx < len(data_df.columns):
                rfu_col = data_df.columns[rfu_col_idx]
                sd_col = data_df.columns[sd_col_idx]
                n_col = data_df.columns[n_col_idx]
                
                # 데이터 추출
                rfu_value = data_df.iloc[idx, rfu_col_idx]
                sd_value = data_df.iloc[idx, sd_col_idx]
                n_value_actual = data_df.iloc[idx, n_col_idx]
                
                # NaN 처리
                if pd.isna(rfu_value):
                    rfu_str = '0'
                else:
                    rfu_str = str(rfu_value)
                
                if pd.isna(sd_value):
                    sd_str = '0'
                else:
                    sd_str = str(sd_value)
                
                # N 값: 원본에서 읽은 값 사용, 없으면 기본값 사용
                if pd.isna(n_value_actual):
                    n_str = str(n_value)
                else:
                    # 숫자로 변환 시도
                    try:
                        n_num = int(float(n_value_actual))
                        n_str = str(n_num)
                    except:
                        n_str = str(n_value)
                
                data_line.extend([rfu_str, sd_str, n_str])
            else:
                # 컬럼이 없는 경우 기본값 사용
                data_line.extend(['0', '0', str(n_value)])
        
        output_lines.append('\t'.join(data_line))
    
    # 출력 경로 설정
    if output_path is None:
        # mode_prep_raw_data 폴더 생성
        os.makedirs('mode_prep_raw_data', exist_ok=True)
        output_path = 'mode_prep_raw_data/raw.csv'
    
    # CSV 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✅ 변환 완료: {output_path}")
    print(f"   총 {len(output_lines) - 2}개 데이터 행 생성")
    print(f"   농도 조건: {len(concentrations)}개")
    
    return output_path


def main():
    """메인 함수"""
    import argparse
    
    # 스크립트가 있는 디렉토리 경로
    script_dir = Path(__file__).parent.absolute()
    default_input = script_dir / 'raw.xlsx'
    
    parser = argparse.ArgumentParser(description='XLSX 파일을 prep_raw.csv 형식으로 변환')
    parser.add_argument('input_file', nargs='?', default=str(default_input),
                       help=f'입력 XLSX 파일 경로 (기본값: {default_input})')
    parser.add_argument('-o', '--output', default=None,
                       help='출력 CSV 파일 경로 (기본값: mode_prep_raw_data/raw.csv)')
    parser.add_argument('-n', '--n-value', type=int, default=50,
                       help='N 값 (복제수, 기본값: 50)')
    
    args = parser.parse_args()
    
    # 입력 파일 경로 처리 (상대 경로인 경우 스크립트 디렉토리 기준으로 변환)
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    
    # 입력 파일 확인
    if not input_path.exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    # 절대 경로로 변환
    args.input_file = str(input_path)
    
    # 변환 실행
    try:
        output_path = convert_xlsx_to_prep_raw(
            args.input_file,
            output_path=args.output,
            n_value=args.n_value
        )
        print(f"\n✨ 변환 성공!")
        print(f"   입력: {args.input_file}")
        print(f"   출력: {output_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

