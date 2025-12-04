# 정규화 과정 플로우차트

```mermaid
flowchart TD
    Start([원본 데이터<br/>F, t]) --> Sort[시간 순서대로 정렬]
    
    Sort --> Init[초기화<br/>current_values = F<br/>iteration = 0]
    
    Init --> Loop{반복 횟수<br/>>= 2?}
    
    Loop -->|Yes| Iter1[반복 시작<br/>iteration++]
    Loop -->|No| Iter1
    
    Iter1 --> Step1[1차 정규화<br/>임시 정규화]
    
    Step1 --> CalcF0["F₀ 계산<br/>F₀ = F at t=0<br/>F₀ = current_values 첫번째값"]
    CalcF0 --> CalcFmax["Fmax 계산<br/>Fmax = max F<br/>Fmax = max current_values"]
    
    CalcFmax --> CalcAlpha["α_temp 계산<br/>α_temp = F - F₀ / Fmax - F₀<br/>normalized = current_values - F₀ / Fmax - F₀"]
    
    CalcAlpha --> Step2[2차 정규화<br/>Exponential 피팅]
    
    Step2 --> FitExp["지수 함수 피팅<br/>F_norm t = F_max · 1 - exp -k_obs · t<br/>정규화된 데이터에 피팅"]
    
    FitExp --> GetParams[파라미터 추출<br/>F_max_fit, k_obs_fit]
    
    GetParams --> CalcTau["특성 시간 계산<br/>τ = 1 / k_obs"]
    
    CalcTau --> CalcV0["초기 속도 계산<br/>v₀ = k_obs · Fmax - F₀<br/>단위: RFU/min"]
    
    CalcV0 --> CheckIter{마지막<br/>반복?}
    
    CheckIter -->|No| Denorm["역정규화<br/>원본 데이터로 변환<br/>F t = F₀ + Fmax - F₀ · 1 - exp -k_obs · t<br/>current_values = fit_values · Fmax - F₀ + F₀"]
    
    Denorm --> Loop
    
    CheckIter -->|Yes| Final[최종 결과]
    
    Final --> Result1[정규화된 데이터<br/>normalized_values]
    Final --> Result2[파라미터<br/>F₀, Fmax, k_obs]
    Final --> Result3["계산값<br/>τ = 1/k_obs<br/>v₀ = k_obs · Fmax - F₀"]
    Final --> Result4["원본 데이터 식<br/>F t = F₀ + Fmax - F₀ · 1 - exp -k_obs · t"]
    
    Result1 --> End([완료])
    Result2 --> End
    Result3 --> End
    Result4 --> End
    
    style Start fill:#e1f5ff
    style End fill:#d4edda
    style Step1 fill:#fff3cd
    style Step2 fill:#ffeaa7
    style FitExp fill:#fdcb6e
    style CalcV0 fill:#6c5ce7,color:#fff
    style CalcTau fill:#6c5ce7,color:#fff
    style Loop fill:#a29bfe,color:#fff
    style CheckIter fill:#a29bfe,color:#fff
```

## 정규화 과정 상세 설명

### 1차 정규화 (임시 정규화)
- **F₀**: 시간 0에서의 형광값 (F(t=0))
- **Fmax**: 최대 형광값 (max(F))
- **정규화**: α_temp = (F - F₀) / (Fmax - F₀)

### 2차 정규화 (Exponential 피팅)
- **정규화된 데이터 식**: F_norm(t) = F_max · [1 - exp(-k_obs · t)]
  - F_norm(t): 정규화된 형광값 (0~1 범위)
  - F_max: 정규화된 최대값 (보통 1.0)
  - k_obs: 관찰된 반응 속도 상수 (분⁻¹)
  - t: 시간 (분)

### 원본 데이터로 변환
- **원본 데이터 식**: F(t) = F₀ + (Fmax - F₀) · [1 - exp(-k_obs · t)]
  - F(t): 시간 t에서의 형광값
  - F₀: 초기 형광값
  - Fmax: 최대 형광값
  - k_obs: 관찰된 반응 속도 상수 (분⁻¹)
  - t: 시간 (분)

### 초기 속도 (v₀) 계산
- **v₀ = k_obs · (Fmax - F₀)**
- 단위: RFU/min (형광 단위/분)

### 특성 시간 (τ)
- **τ = 1 / k_obs**
- 반응이 63.2% 완료되는 시간

### 반복 정규화
- 위 과정을 최소 2번 반복하여 정규화를 개선
- 각 반복에서 피팅된 값을 역정규화하여 다음 반복에 사용

