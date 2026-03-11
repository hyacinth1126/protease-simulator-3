# 플롯별 사용 식 정리

저장되는 각 플롯에서 쓰이는 수식을 정리한 문서입니다.

---

## 파라미터 정의 (논문용)

본 문서의 식에 등장하는 기호는 아래와 같이 정의하며, Methods 또는 기호 설명란에 그대로 인용할 수 있다.

| 기호 | 의미 (논문용 설명) |
|------|---------------------|
| $F(t)$ | **시간 $t$에서의 형광 강도** (RFU). 반응 진행에 따라 변하는 형광 신호. |
| $F_0$ | **초기 형광** (baseline fluorescence). $t=0$에서의 형광 강도로, 기질만 있을 때의 신호 또는 반응 시작 직후의 값을 나타낸다. |
| $F_{\max}$ | **포화 형광** (plateau / maximum fluorescence). 반응이 완료에 가깝게 진행된 후 도달하는 최대 형광 강도. 기질 소진 또는 평형에 해당한다. |
| $F_\infty$ | **완전 절단 시 형광** (fluorescence at full cleavage). 절단 반응이 100% 진행되었을 때의 형광; 농도별로 하나의 값이거나, 공통 $F_\infty$를 쓸 수 있다. |
| $F_t$ | **시간 $t$에서의 측정 형광**. $\alpha(t)$ 계산 시 해당 시점의 관측값. |
| $F_{\mathrm{norm}}$ | **정규화 형광**. $(F - F_0)/(F_{\max} - F_0)$로, 0~1 범위의 비율(진행도)을 나타낸다. |
| $k$ | **유사 1차 속도 상수** (apparent first-order rate constant, min$^{-1}$). 지수 포화 곡선 $F(t) = F_0 + (F_{\max}-F_0)(1-e^{-kt})$에서 포화에 도달하는 속도를 나타낸다. |
| $k_{\mathrm{obs}}$ | **관측 속도 상수** (observed rate constant, min$^{-1}$). 정규화된 곡선 또는 progress curve 피팅에서 얻은 지수 상수. $k_{\mathrm{obs}} = v_0/(F_{\max}-F_0)$ 관계가 성립할 수 있다. |
| $\tau$ | **시간 상수** (time constant). $\tau = 1/k_{\mathrm{obs}}$로, 반응이 포화의 약 63%에 도달하는 데 걸리는 시간. |
| $t$ | **시간** (min 또는 s). 반응 시작($t=0$)으로부터의 경과 시간. |
| $v_0$ | **초기 속도** (initial velocity). 반응 초기 구간에서의 형광 증가율 (RFU/min 등). 선형 구간 기울기 또는 $v_0 = k_{\mathrm{obs}}(F_{\max}-F_0)$로 구한다. |
| $V_{\max}$ | **최대 속도** (maximum velocity). Michaelis–Menten 식에서 기질 농도가 포화될 때의 반응 속도 (같은 단위 as $v_0$). |
| $K_m$ | **Michaelis 상수** (Michaelis constant). $v_0 = V_{\max}/2$가 되는 기질 농도; 효소–기질 친화성과 관련된다. |
| $[S]$ | **기질 농도** (substrate concentration, μM 등). |
| $[E]$ | **효소 농도** (enzyme concentration, M, μM, μg/mL 등). |
| $\alpha(t)$ | **절단 비율** (fraction cleaved). $\alpha = (F_t - F_0)/(F_\infty - F_0)$로, 0(미절단)~1(완전 절단) 사이의 진행도. |
| $\Gamma$, $\Gamma_0$, $\Gamma_t$ | **기질(또는 잔여 기질) 양**. Model A 전역 피팅에서 사용; $\alpha = 1 - \Gamma_t/\Gamma_0$. |
| $k_{\mathrm{cat}}/K_M$ | **특이성 상수** (specificity constant). 효소 효율의 지표; 단순 1차 근사에서 $k_{\mathrm{obs}} = (k_{\mathrm{cat}}/K_M)[E]$. |

**지수 포화 곡선**  
$F(t) = F_0 + (F_{\max} - F_0)(1 - e^{-kt})$는 단일 지수 결합(exponential association) 모델로, 형광이 초기값 $F_0$에서 포화값 $F_{\max}$로 시간에 따라 증가하며 포화에 도달하는 과정을 나타낸다. $k$는 그 속도를 결정하는 유사 1차 속도 상수이다.

**Michaelis–Menten**  
$v_0 = V_{\max}[S]/(K_m + [S])$는 기질 농도에 따른 초기 속도를 나타내는 보정 곡선이며, $V_{\max}$와 $K_m$은 비선형 회귀로 추정한다.

---

## 1. Data Load 모드 — Export 플롯

### 1.1 Experimental_Results

- **설명**: 원시 데이터만 표시 (Time vs RFU). 곡선 식 없음.
- **축**: x = Time (min), y = RFU

---

### 1.2 Time_Fluorescence_Interpolated_Curves

- **설명**: 농도별 Time–RFU 보간 곡선 (지수 포화).
- **곡선 식**

$$
F(t) = F_0 + (F_{\max} - F_0)(1 - e^{-k t})
$$

- **Substrate 농도 변화**: $k = v_0 / (F_{\max} - F_0)$. 정규화 후에는 $k_{\mathrm{obs}}$ 사용.
- **Enzyme 농도 변화**: 선형 $F(t) = F_0 + v_0 t$.
- **기호**: $F_0$ = t=0 형광, $F_{\max}$ = 포화 형광, $k$ = 속도 상수, $t$ = 시간 (min).

---

### 1.3 Normalized_Time_Fluorescence_exponential_curves_3tau_max_range

- **설명**: 정규화된 형광 곡선 (0~1), 지수 구간 강조, 최대 3τ 범위.
- **정규화**

$$
F_{\mathrm{norm}} = \frac{F - F_0}{F_{\max} - F_0}
$$

- **피팅 곡선 (정규화 공간, $F_{\max}=1$)**
$$
F_{\mathrm{norm}}(t) = F_{\max}(1 - e^{-k_{\mathrm{obs}} t}) = 1 - e^{-k_{\mathrm{obs}} t}
$$

- **초기 선형 구간 (대시선)**
$$
y = k_{\mathrm{obs}} \, t
$$

- **시간 상수**: $\tau = 1 / k_{\mathrm{obs}}$.

---

### 1.4 v0_vs_S_Fit (Initial Velocity vs Substrate Concentration)

- **설명**: v₀ vs [S] Michaelis–Menten 보정 곡선.
- **곡선 식**

$$
v_0 = \frac{V_{\max} \cdot [S]}{K_m + [S]}
$$

- **기호**: $[S]$ = 기질 농도 (μM), $V_{\max}$, $K_m$ = 피팅 파라미터.

---

### 1.5 Linear_fit (v₀ vs Enzyme Concentration)

- **설명**: 기질 고정 시 v₀ vs [E] 선형 피팅.
- **곡선 식**

$$
v_0 = \mathrm{slope} \cdot [E] + \mathrm{intercept}
$$

---

### 1.6 Supplementary_low3_const_fit

- **설명**: Enzyme 모드에서 낮은 농도 3점만 사용한 보조 선형 피팅 (상수항 포함).
- **곡선 식**: 위와 동일한 선형 $v_0 = a \cdot [E] + b$.

---

### 1.7 Normalization_{농도} (Enzyme-quenched peptide fluorescence kinetics)

- **설명**: 농도별 정규화 데이터 + 지수 피팅 + 초기 선형 구간.
- **정규화 데이터**: $F_{\mathrm{norm}} = (F - F_0)/(F_{\max} - F_0)$.
- **지수 곡선 (주황)**

$$
F_{\mathrm{norm}}(t) = F_{\max}(1 - e^{-k_{\mathrm{obs}} t}), \quad F_{\max}=1
$$

- **초기 선형 (점선)**
$$
y = k_{\mathrm{obs}} \, t
$$

- **v₀ (원시 단위)**: $v_0 = k_{\mathrm{obs}}(F_{\max} - F_0)$.

---

### 1.8 Normalization_to_plateau_{농도}

- **설명**: 위와 동일 식, y≈1(plateau) 도달 구간까지만 표시.

---

## 2. Data Load 모드 — 정규화 피팅 (내부)

- **정규화**: $F_{\mathrm{norm}} = (F - F_0)/(F_{\max} - F_0)$, $F_0$ = 첫 시간점 값, $F_{\max}$ = 최대값.
- **정규화 공간 지수 피팅**

$$
F_{\mathrm{norm}}(t) = F_{\max}(1 - e^{-k_{\mathrm{obs}} t})
$$

($F_{\max}$ 피팅, 보통 ≈1.)
- **역정규화**: $F(t) = F_{\mathrm{norm}}(t)(F_{\max} - F_0) + F_0$.

---

## 3. Prep 모드 — MM Calibration Curve

- **파일**: `prep_data/fitting_results/MM_calibration_curve.png`
- **설명**: v₀ vs [S] 보정 곡선.
- **곡선 식**

$$
v_0 = \frac{V_{\max} \cdot x}{K_m + x}
$$

$x$ = 기질 농도, $V_{\max}$, $K_m$ = 피팅 파라미터.

---

## 4. Data Interpolation Mode (standalone)

- **파일**: `MM_interpolated_curves.png` (interpolate_prism.py)
- **곡선 식**: Data Load와 동일한 지수 포화

$$
F(t) = F_0 + (F_{\max} - F_0)(1 - e^{-k t})
$$

$F_0$, $F_{\max}$, $k$ = MM_results_detailed.csv 또는 피팅 결과.

---

## 5. Model Simulation (General Analysis) — α 및 모델

### 5.1 절단 비율 α(t)

$$
\alpha(t) = \frac{F_t - F_0}{F_\infty - F_0}
$$

- $F_t$: 시간 $t$에서 형광, $F_0$: 초기 형광, $F_\infty$: 완전 절단 시 형광. $\alpha \in [0, 1]$로 클리핑.

---

### 5.2 Progress curve 피팅 (형광 F vs t)

- **3-파라미터**

$$
F(t) = F_0 + (F_\infty - F_0)(1 - e^{-k_{\mathrm{obs}} t})
$$

- **2-파라미터** ($F_0=0$ 가정)

$$
F(t) = F_\infty(1 - e^{-k_{\mathrm{obs}} t})
$$

---

### 5.3 Model A: Substrate Depletion (단순 1차)

$$
\alpha(t) = 1 - e^{-k_{\mathrm{obs}} t}, \qquad k_{\mathrm{obs}} = (k_{\mathrm{cat}}/K_M) \cdot [E]
$$

- $[E]$: 효소 농도 (M 등), $k_{\mathrm{cat}}/K_M$ 피팅.

---

### 5.4 Model A: Full (Michaelis–Menten, 수치 적분)

- $d\Gamma/dt = -v$, $v = k_{\mathrm{cat}} [E] \Gamma / (K_M + \Gamma)$.
- $\alpha = 1 - \Gamma_t / \Gamma_0$, $\alpha \in [0, 1]$ 클리핑.

---

## 요약 표

| 플롯/출력 | 사용 식 |
|-----------|---------|
| Time-Fluorescence Interpolated | $F(t) = F_0 + (F_{\max}-F_0)(1 - e^{-kt})$ 또는 $F_0 + v_0 t$ |
| Normalized exponential curves | $F_{\mathrm{norm}} = 1 - e^{-k_{\mathrm{obs}} t}$, $y = k_{\mathrm{obs}} t$ |
| v₀ vs [S] | $v_0 = V_{\max}[S]/(K_m + [S])$ |
| v₀ vs [E] | $v_0 = \mathrm{slope}\cdot [E] + \mathrm{intercept}$ |
| Normalization 탭 | $F_{\mathrm{norm}} = (F-F_0)/(F_{\max}-F_0)$, $1 - e^{-k_{\mathrm{obs}} t}$ |
| MM calibration (prep) | $v_0 = V_{\max} x/(K_m + x)$ |
| α(t) (Model Simulation) | $\alpha = (F_t - F_0)/(F_\infty - F_0)$ |
| Model A 단순 | $\alpha = 1 - e^{-k_{\mathrm{obs}} t}$ |
