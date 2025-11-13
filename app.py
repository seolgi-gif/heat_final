import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 한글 폰트 설정 (Streamlit Cloud 최적화) ---
@st.cache_data
def font_setup():
    fm._rebuild()
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic_files = [f for f in font_files if 'NanumGothic' in f]
    if nanum_gothic_files:
        plt.rc('font', family='NanumGothic')
        font_prop = fm.FontProperties(fname=nanum_gothic_files[0])
    else:
        st.warning("나눔고딕 폰트를 찾을 수 없습니다. 글자가 깨질 수 있습니다.")
        font_prop = fm.FontProperties(size=12)
    plt.rcParams['axes.unicode_minus'] = False
    return font_prop

font_prop = font_setup()

# --- 2. 시나리오(재료) 정의 ---
# ※ 참고: PCM은 상변화(잠열) 효과가 반영되지 않은 고체 상태의 물성치입니다.
SCENARIOS = {
    '에어로겔': {'k': 0.02, 'rho': 80, 'cp': 1000},
    '세라믹 섬유': {'k': 0.1, 'rho': 150, 'cp': 1000},
    'PCM (고체상태)': {'k': 0.25, 'rho': 900, 'cp': 2100},
    '강철 (Steel)': {'k': 50.0, 'rho': 7850, 'cp': 490},
    '알루미늄': {'k': 200.0, 'rho': 2700, 'cp': 900},
}

# --- 3. 다층 구조 2D 열전달 시뮬레이션 함수 ---
def run_multilayer_simulation(materials, thicknesses_m, T_hot_c=1000, T_initial_c=20, T_target_c=120, sim_time_minutes=15):
    """
    여러 층으로 구성된 재료의 2D 열전달을 시뮬레이션합니다.
    materials: 각 층의 재료 속성 딕셔너리 리스트
    thicknesses_m: 각 층의 두께(m) 리스트
    """
    T_hot = T_hot_c + 273.15
    T_initial = T_initial_c + 273.15
    T_target_kelvin = T_target_c + 273.15
    sim_time_seconds = sim_time_minutes * 60
    
    L_x = sum(thicknesses_m)
    L_y = 0.1
    nx, ny = 100, 10  # 두께 방향 격자를 더 세밀하게 설정
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)

    # 각 격자점(node)에 해당하는 재료의 열확산계수(alpha) 맵 생성
    alpha_map = np.zeros(nx)
    alphas = [mat['k'] / (mat['rho'] * mat['cp']) for mat in materials]
    
    current_pos_m = 0
    start_idx = 0
    for i, thick_m in enumerate(thicknesses_m):
        current_pos_m += thick_m
        end_idx = int(current_pos_m / L_x * (nx - 1))
        alpha_map[start_idx : end_idx + 1] = alphas[i]
        start_idx = end_idx

    # 안정성 조건을 위해 가장 큰 alpha 값을 기준으로 dt 계산
    max_alpha = max(alphas)
    dt = 0.2 * (1 / (max_alpha * (1/dx**2 + 1/dy**2)))
    if dt > 0.5: dt = 0.5
    nt = int(sim_time_seconds / dt)
    if nt <= 0: return None, None, None, None

    # 시뮬레이션 초기화
    time_points = np.linspace(0, sim_time_seconds, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial
    time_to_target = None

    # 시간 반복 계산
    for t_step in range(nt):
        T_old = T.copy()
        T[:, 0] = T_hot; T[:, -1] = T[:, -2]; T[0, :] = T[1, :]; T[-1, :] = T[-2, :]
        
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                term1 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dy**2
                term2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dx**2
                T[i, j] = T_old[i, j] + alpha_map[j] * dt * (term1 + term2)
        
        current_inner_temp_k = np.mean(T[:, -1])
        temp_history_celsius[t_step] = current_inner_temp_k - 273.15
        
        if time_to_target is None and current_inner_temp_k >= T_target_kelvin:
            time_to_target = time_points[t_step] / 60
            
    return time_points, temp_history_celsius, T - 273.15, time_to_target

# --- 4. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("🚗 자동차 배터리 열차폐 시스템 설계 시뮬레이션")
st.markdown("""
이 앱은 자동차 배터리 팩을 외부 고온(1000°C)으로부터 보호하기 위한 열차폐 시스템을 설계하는 데 도움을 줍니다.
1.  **최대 허용 두께**와 **목표 지연 시간**을 설정합니다.
2.  **단일 재료 분석**을 통해 각 재료의 기본 성능을 확인합니다.
3.  결과를 바탕으로 **최적의 다층 구조**를 설계하고 성능을 검증합니다.
""")

# --- 4.1. 시뮬레이션 조건 설정 (사이드바) ---
st.sidebar.header("⚙️ 1. 기본 조건 설정")
max_thickness_mm = st.sidebar.number_input("최대 허용 두께 (mm)", min_value=5.0, max_value=100.0, value=30.0, step=1.0)
target_delay_min = st.sidebar.number_input("목표 지연 시간 (분)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)

# --- 4.2. 단일 재료 성능 분석 ---
st.header("📊 1단계: 단일 재료 성능 분석")
st.markdown(f"각 재료를 **{max_thickness_mm}mm** 두께로 단독 사용했을 때, 내부 온도가 120°C에 도달하는 시간을 계산합니다.")

if st.button("단일 재료 분석 시작"):
    results = []
    with st.spinner("각 재료의 성능을 분석 중입니다..."):
        for name, props in SCENARIOS.items():
            _, _, _, time_to_target = run_multilayer_simulation(
                materials=[props],
                thicknesses_m=[max_thickness_mm / 1000.0],
                sim_time_minutes=target_delay_min * 3 # 충분한 시뮬레이션 시간 확보
            )
            
            if time_to_target is None:
                delay_str = f"{target_delay_min * 3}분 이상"
                is_success = True
            else:
                delay_str = f"{time_to_target:.2f} 분"
                is_success = time_to_target >= target_delay_min

            results.append({
                "재료": name,
                "120°C 도달 시간": delay_str,
                f"목표({target_delay_min}분) 달성": "✅" if is_success else "❌"
            })
    
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.success("분석이 완료되었습니다. 위 결과를 바탕으로 아래에서 다층 구조를 설계하세요.")

# --- 4.3. 다층 구조 설계 및 시뮬레이션 ---
st.header("🛠️ 2단계: 다층 구조 설계 및 시뮬레이션")
st.markdown("1단계 분석 결과를 바탕으로, 목표를 달성할 가능성이 높은 재료 3개를 조합하여 최적의 구조를 찾아보세요.")

# 다층 구조 설계를 위한 세션 상태 초기화
if 'layer_thicknesses' not in st.session_state:
    st.session_state.layer_thicknesses = [max_thickness_mm / 3] * 3

# 재료 선택
material_options = list(SCENARIOS.keys())
selected_materials = st.multiselect(
    "3개의 재료를 선택하세요 (외부 -> 내부 순서)",
    options=material_options,
    default=['세라믹 섬유', 'PCM (고체상태)', '에어로겔'],
    max_selections=3
)

if len(selected_materials) == 3:
    st.subheader("두께 분배")
    cols = st.columns(3)
    thicknesses = []
    for i, mat_name in enumerate(selected_materials):
        with cols[i]:
            thickness = st.slider(f"Layer {i+1}: {mat_name} (mm)", 0.0, max_thickness_mm, st.session_state.layer_thicknesses[i], 0.5, key=f"thick_{i}")
            thicknesses.append(thickness)

    total_selected_thickness = sum(thicknesses)
    if total_selected_thickness > max_thickness_mm:
        st.error(f"선택한 두께의 총합({total_selected_thickness:.1f}mm)이 최대 허용 두께({max_thickness_mm}mm)를 초과했습니다.")
    else:
        st.info(f"현재 총 두께: {total_selected_thickness:.1f} mm / {max_thickness_mm} mm")

    if st.button("다층 구조 시뮬레이션 실행", key="run_multilayer"):
        with st.spinner("다층 구조 시뮬레이션을 진행 중입니다..."):
            
            materials_to_sim = [SCENARIOS[name] for name in selected_materials]
            thicknesses_to_sim_m = [t / 1000.0 for t in thicknesses]

            time_pts, temp_hist, final_temp_dist, time_to_target = run_multilayer_simulation(
                materials=materials_to_sim,
                thicknesses_m=thicknesses_to_sim_m,
                sim_time_minutes=target_delay_min * 1.5
            )

        st.subheader("🚀 시뮬레이션 결과")
        final_delay = time_to_target if time_to_target is not None else (target_delay_min * 1.5)
        
        c1, c2 = st.columns(2)
        c1.metric("120°C 도달 시간", f"{final_delay:.2f} 분" if time_to_target else f"{target_delay_min*1.5}분 이상")
        if final_delay >= target_delay_min:
            c2.metric("목표 지연 시간 달성 여부", "✅ 성공")
        else:
            c2.metric("목표 지연 시간 달성 여부", "❌ 실패")

        # --- 결과 시각화 ---
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(time_pts / 60, temp_hist, label=f"다층 구조 내부 온도", lw=2.5)
        ax1.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
        ax1.axvline(x=target_delay_min, color='g', linestyle=':', label=f'목표 지연 시간 ({target_delay_min}분)')
        
        ax1.set_title(f'내부 표면 온도 변화', fontproperties=font_prop, fontsize=16)
        ax1.set_xlabel('시간 (분)', fontproperties=font_prop)
        ax1.set_ylabel('온도 (°C)', fontproperties=font_prop)
        ax1.legend(prop=font_prop); ax1.grid(True, linestyle=':'); 
        ax1.set_xlim(0, target_delay_min * 1.5)
        max_temp_visual = max(temp_hist)
        ax1.set_ylim(15, max(150, max_temp_visual * 1.2))
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 2))
        im = ax2.imshow(final_temp_dist, cmap='inferno', aspect='auto', extent=[0, sum(thicknesses), 0, 10], vmin=20, vmax=1000)
        fig2.colorbar(im, ax=ax2, label='온도 (°C)')
        ax2.set_title(f'최종 시간에서의 2D 온도 분포', fontproperties=font_prop, fontsize=16)
        ax2.set_xlabel('두께 방향 (mm)', fontproperties=font_prop)
        ax2.set_yticks([]) # y축 눈금 제거
        
        # 재료 경계선 표시
        current_pos = 0
        for i, thick in enumerate(thicknesses[:-1]):
            current_pos += thick
            ax2.axvline(x=current_pos, color='white', linestyle='--', lw=1.5)
            ax2.text(current_pos - thick/2, 5, selected_materials[i], color='white', ha='center', va='center', fontproperties=font_prop)
        ax2.text(current_pos + thicknesses[-1]/2, 5, selected_materials[-1], color='white', ha='center', va='center', fontproperties=font_prop)
        
        st.pyplot(fig2)

else:
    st.warning("먼저 3개의 재료를 선택해주세요.")
