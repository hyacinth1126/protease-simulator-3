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
    print(f"Reading XLSX: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name='Sheet1', engine='openpyxl', header=None)
    
    print(f"   Size: {df.shape[0]} rows x {df.shape[1]} cols")
    
    # 첫 번째 행: 컬럼 헤더 (min, 농도들)
    first_row = df.iloc[0].values
    time_col_name = first_row[0]  # 'min' 또는 'time_min'
    
    # 두 번째 행: 헤더 타입 (RFU, SD, N) 또는 (RFU, SD) 만
    second_row = df.iloc[1].values
    n_data_cols = len(first_row) - 1  # time 제외
    
    # 형식 감지: Substrate = 농도당 3열(RFU, SD, N), Enzyme = 농도당 2열(RFU, SD, N 없음)
    has_n_column = any(
        pd.notna(v) and 'N' in str(v).upper()
        for v in second_row[1:]
    )
    if n_data_cols % 3 == 0 and has_n_column:
        cols_per_conc = 3
        format_type = "substrate"
    elif n_data_cols % 2 == 0:
        cols_per_conc = 2
        format_type = "enzyme"
    else:
        cols_per_conc = 3
        format_type = "substrate"
    
    # 농도 값 추출
    concentrations = []
    for i in range(1, len(first_row), cols_per_conc):
        if i < len(first_row):
            conc_value = extract_concentration_from_header(first_row[i])
            if conc_value is not None:
                concentrations.append(conc_value)
    
    # enzyme: 농도가 홀수 인덱스에만 있는 경우 (0.3125ug/ml, nan, 0.625 ug/mL, nan, ...)
    if not concentrations and cols_per_conc == 2:
        for i in range(1, len(first_row), 2):
            if i < len(first_row):
                conc_value = extract_concentration_from_header(first_row[i])
                if conc_value is not None:
                    concentrations.append(conc_value)
    
    print(f"   Format: {format_type}, Concentrations: {concentrations}")
    
    # 세 번째 행부터: 실제 데이터 (컬럼명 없이 인덱스로 접근)
    data_df = df.iloc[2:].copy()
    n_cols = data_df.shape[1]
    
    # 시간 컬럼 (인덱스 0)
    times = pd.to_numeric(data_df.iloc[:, 0].values, errors='coerce')
    
    # prep_raw.csv 형식으로 변환
    output_lines = []
    
    # 첫 번째 행: 농도 값들 (각 농도가 mean, SD, N으로 3번 반복)
    first_line = ['']
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
        
        for i in range(len(concentrations)):
            if cols_per_conc == 3:
                # Substrate: RFU, SD, N 3열
                rfu_col_idx = 1 + i * 3
                sd_col_idx = 2 + i * 3
                n_col_idx = 3 + i * 3
                if rfu_col_idx < n_cols and sd_col_idx < n_cols and n_col_idx < n_cols:
                    rfu_value = data_df.iloc[idx, rfu_col_idx]
                    sd_value = data_df.iloc[idx, sd_col_idx]
                    n_value_actual = data_df.iloc[idx, n_col_idx]
                else:
                    rfu_value = sd_value = n_value_actual = None
                rfu_str = '0' if pd.isna(rfu_value) else str(rfu_value)
                sd_str = '0' if pd.isna(sd_value) else str(sd_value)
                if pd.isna(n_value_actual):
                    n_str = str(n_value)
                else:
                    try:
                        n_str = str(int(float(n_value_actual)))
                    except Exception:
                        n_str = str(n_value)
                data_line.extend([rfu_str, sd_str, n_str])
            else:
                # Enzyme: RFU, SD 2열만 있음 -> N을 기본값으로 끼워 넣음
                rfu_col_idx = 1 + i * 2
                sd_col_idx = 2 + i * 2
                if rfu_col_idx < n_cols and sd_col_idx < n_cols:
                    rfu_value = data_df.iloc[idx, rfu_col_idx]
                    sd_value = data_df.iloc[idx, sd_col_idx]
                else:
                    rfu_value = sd_value = None
                rfu_str = '0' if pd.isna(rfu_value) else str(rfu_value)
                sd_str = '0' if pd.isna(sd_value) else str(sd_value)
                n_str = str(n_value)
                data_line.extend([rfu_str, sd_str, n_str])
        
        output_lines.append('\t'.join(data_line))
    
    # 출력 경로 설정
    if output_path is None:
        # mode_prep_raw_data 폴더 생성
        os.makedirs('mode_prep_raw_data', exist_ok=True)
        output_path = 'mode_prep_raw_data/raw.csv'
    
    # CSV 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"OK: {output_path}")
    print(f"   Data rows: {len(output_lines) - 2}, concentrations: {len(concentrations)}")
    
    return output_path


def main():
    """메인 함수"""
    import argparse
    
    # 스크립트가 있는 디렉토리 경로
    script_dir = Path(__file__).parent.absolute()
    raw_dir = script_dir / 'raw'
    csv_dir = script_dir / 'csv'
    
    parser = argparse.ArgumentParser(description='XLSX 파일을 prep_raw.csv 형식으로 변환')
    parser.add_argument('input_file', nargs='?', default=None,
                       help='입력 XLSX 파일 경로 (생략 시 raw/ 폴더 내 모든 XLSX 변환)')
    parser.add_argument('-o', '--output', default=None,
                       help='출력 CSV 파일 경로 (단일 파일 모드에서만 사용)')
    parser.add_argument('-n', '--n-value', type=int, default=50,
                       help='N 값 (복제수, 기본값: 50)')
    
    args = parser.parse_args()
    
    # 입력 생략 시: raw/ 내 모든 XLSX → csv/ 폴더에 저장
    if args.input_file is None or args.input_file == '':
        if not raw_dir.exists():
            print(f"오류: raw 폴더를 찾을 수 없습니다: {raw_dir}")
            sys.exit(1)
        xlsx_files = sorted(raw_dir.glob('*.xlsx'))
        if not xlsx_files:
            print(f"오류: raw 폴더에 XLSX 파일이 없습니다: {raw_dir}")
            sys.exit(1)
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"raw/ 폴더 내 XLSX {len(xlsx_files)}개 -> csv/ 폴더로 변환\n")
        success = 0
        for xlsx_path in xlsx_files:
            try:
                out_path = csv_dir / (xlsx_path.stem + '.csv')
                convert_xlsx_to_prep_raw(
                    str(xlsx_path),
                    output_path=str(out_path),
                    n_value=args.n_value
                )
                success += 1
            except Exception as e:
                print(f"변환 실패 ({xlsx_path.name}): {e}")
                import traceback
                traceback.print_exc()
        print(f"\n완료: {success}/{len(xlsx_files)}개 파일 -> {csv_dir}")
        return
    
    # 단일 파일 모드
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    
    if not input_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    args.input_file = str(input_path)
    output_path = args.output
    if output_path is None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(csv_dir / (input_path.stem + '.csv'))
    
    try:
        result_path = convert_xlsx_to_prep_raw(
            args.input_file,
            output_path=output_path,
            n_value=args.n_value
        )
        print(f"\n변환 성공!")
        print(f"   입력: {args.input_file}")
        print(f"   출력: {result_path}")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

