import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 한글 폰트 설정 (이전과 동일) ---
@st.cache_data
def font_setup():
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
SCENARIOS = {
    '에어로겔': {'k': 0.02, 'rho': 80, 'cp': 1000},
    '세라믹 섬유': {'k': 0.1, 'rho': 150, 'cp': 1000},
    'PCM (고체상태)': {'k': 0.25, 'rho': 900, 'cp': 2100},
    '강철 (Steel)': {'k': 50.0, 'rho': 7850, 'cp': 490},
    '알루미늄': {'k': 200.0, 'rho': 2700, 'cp': 900},
}

# --- 3. 최적화 및 오류 수정된 시뮬레이션 함수 ---
@st.cache_data
def run_multilayer_simulation(materials, thicknesses_m, T_hot_c=1000, T_initial_c=20, T_target_c=120, sim_time_minutes=15, stop_at_target=False):
    T_hot = T_hot_c + 273.15
    T_initial = T_initial_c + 273.15
    T_target_kelvin = T_target_c + 273.15
    sim_time_seconds = sim_time_minutes * 60
    
    L_x = sum(thicknesses_m)
    if L_x == 0: return None, None, None, None
    L_y = 0.1
    
    nx, ny = 60, 6
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)

    alpha_map = np.zeros(nx)
    alphas = [mat['k'] / (mat['rho'] * mat['cp']) for mat in materials]
    
    current_pos_m = 0
    start_idx = 0
    for i, thick_m in enumerate(thicknesses_m):
        current_pos_m += thick_m
        end_idx = int(current_pos_m / L_x * (nx - 1))
        alpha_map[start_idx : end_idx + 1] = alphas[i]
        start_idx = end_idx

    max_alpha = max(alphas)
    dt = 0.2 * (1 / (max_alpha * (1/dx**2 + 1/dy**2)))
    if dt > 0.5: dt = 0.5
    nt = int(sim_time_seconds / dt)
    if nt <= 0: return None, None, None, None

    time_points = np.linspace(0, sim_time_seconds, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial
    time_to_target = None

    for t_step in range(nt):
        T_old = T.copy()
        
        # === 핵심 오류 수정 부분 ===
        # 이전 시간(T_old)을 기준으로 Laplacian(온도 변화율) 계산
        laplacian_x = (T_old[1:-1, 2:] - 2 * T_old[1:-1, 1:-1] + T_old[1:-1, :-2]) / dx**2
        laplacian_y = (T_old[2:, 1:-1] - 2 * T_old[1:-1, 1:-1] + T_old[:-2, 1:-1]) / dy**2
        
        alpha_slice = alpha_map[1:-1]
        
        # 계산된 변화량을 T_old에 더하여 다음 시간(T)의 온도를 계산
        change_in_T = alpha_slice * dt * (laplacian_x + laplacian_y)
        T[1:-1, 1:-1] = T_old[1:-1, 1:-1] + change_in_T
        # ==========================

        # 경계 조건 적용
        T[:, 0] = T_hot; T[:, -1] = T[:, -2]; T[0, :] = T[1, :]; T[-1, :] = T[-2, :]
        
        current_inner_temp_k = np.mean(T[:, -1])
        temp_history_celsius[t_step] = current_inner_temp_k - 273.15
        
        if time_to_target is None and current_inner_temp_k >= T_target_kelvin:
            time_to_target = time_points[t_step] / 60
            if stop_at_target:
                return time_points[:t_step+1], temp_history_celsius[:t_step+1], T - 273.15, time_to_target
            
    return time_points, temp_history_celsius, T - 273.15, time_to_target

# --- 4. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("🚗 자동차 배터리 열차폐 시스템 설계 시뮬레이션")
st.markdown("외부 1000°C 화염 조건에서, 설정된 두께와 재료 조합에 따른 배터리 팩 내부 온도 변화를 예측합니다.")

st.sidebar.header("⚙️ 1. 기본 조건 설정")
max_thickness_mm = st.sidebar.number_input("최대 허용 두께 (mm)", 5.0, 100.0, 30.0, 1.0)
target_delay_min = st.sidebar.number_input("목표 지연 시간 (분)", 1.0, 30.0, 5.0, 0.5)

st.header("📊 1단계: 단일 재료 성능 분석")
st.markdown(f"각 재료를 **{max_thickness_mm}mm** 두께로 단독 사용했을 때, 내부 온도가 120°C에 도달하는 시간을 계산합니다.")

if st.button("단일 재료 분석 시작"):
    results = []
    st.info("각 재료의 성능을 분석 중입니다. 캐싱 기능으로 두 번째 실행부터는 즉시 완료됩니다.")
    progress_bar = st.progress(0, text="분석 시작...")
    
    sorted_scenarios = sorted(SCENARIOS.items(), key=lambda item: item[1]['k']) # 단열 성능 좋은 순으로 정렬

    for i, (name, props) in enumerate(sorted_scenarios):
        progress_bar.progress((i + 1) / len(SCENARIOS), text=f"분석 중: {name}")
        _, _, _, time_to_target = run_multilayer_simulation(
            materials=[(name, props)], # 캐싱을 위해 이름도 함께 전달
            thicknesses_m=[max_thickness_mm / 1000.0],
            sim_time_minutes=target_delay_min * 3,
            stop_at_target=True
        )
        
        delay_str = f"{time_to_target:.2f} 분" if time_to_target else f"{target_delay_min * 3}분 이상"
        is_success = time_to_target is None or time_to_target >= target_delay_min
        results.append({
            "재료": name,
            "120°C 도달 시간": delay_str,
            f"목표({target_delay_min}분) 달성": "✅" if is_success else "❌"
        })
    
    progress_bar.empty()
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.success("분석이 완료되었습니다. 위 결과를 바탕으로 아래에서 다층 구조를 설계하세요.")

st.header("🛠️ 2단계: 다층 구조 설계 및 시뮬레이션")
st.markdown("1단계 분석 결과를 바탕으로, 목표를 달성할 가능성이 높은 재료 3개를 조합하여 최적의 구조를 찾아보세요.")

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
            # UI 개선: 재료가 바뀌면 슬라이더의 기본값도 재설정되도록 key 사용
            thicknesses.append(st.slider(f"Layer {i+1}: {mat_name} (mm)", 0.0, max_thickness_mm, max_thickness_mm / 3, 0.5, key=f"thick_{i}_{mat_name}"))

    total_selected_thickness = sum(thicknesses)
    if total_selected_thickness > max_thickness_mm:
        st.error(f"선택한 두께의 총합({total_selected_thickness:.1f}mm)이 최대 허용 두께({max_thickness_mm}mm)를 초과했습니다.")
    else:
        st.info(f"현재 총 두께: {total_selected_thickness:.1f} mm / {max_thickness_mm} mm")

    if st.button("다층 구조 시뮬레이션 실행", key="run_multilayer"):
        if sum(thicknesses) <= 0:
            st.error("두께를 0보다 크게 설정해야 시뮬레이션이 가능합니다.")
        else:
            with st.spinner("다층 구조 시뮬레이션을 진행 중입니다..."):
                materials_to_sim = [(name, SCENARIOS[name]) for name in selected_materials]
                thicknesses_to_sim_m = [t / 1000.0 for t in thicknesses]
                time_pts, temp_hist, _, time_to_target = run_multilayer_simulation(
                    materials=materials_to_sim,
                    thicknesses_m=thicknesses_to_sim_m,
                    sim_time_minutes=target_delay_min * 1.5
                )

            st.subheader("🚀 시뮬레이션 결과")
            if time_pts is None:
                st.error("시뮬레이션 조건을 계산할 수 없습니다.")
            else:
                final_delay = time_to_target if time_to_target is not None else (target_delay_min * 1.5)
                c1, c2 = st.columns(2)
                c1.metric("120°C 도달 시간", f"{final_delay:.2f} 분" if time_to_target else f"{target_delay_min*1.5}분 이상")
                c2.metric("목표 지연 시간 달성 여부", "✅ 성공" if final_delay >= target_delay_min else "❌ 실패")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(time_pts / 60, temp_hist, label="다층 구조 내부 온도", lw=2.5)
                ax.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
                ax.axvline(x=target_delay_min, color='g', linestyle=':', label=f'목표 지연 시간 ({target_delay_min}분)')
                ax.set_title('내부 표면 온도 변화', fontproperties=font_prop, fontsize=16)
                ax.set_xlabel('시간 (분)', fontproperties=font_prop)
                ax.set_ylabel('온도 (°C)', fontproperties=font_prop)
                ax.legend(prop=font_prop); ax.grid(True, linestyle=':'); ax.set_xlim(0, target_delay_min * 1.5)
                ax.set_ylim(15, max(150, np.max(temp_hist) * 1.2) if len(temp_hist) > 0 else 150)
                st.pyplot(fig)
else:
    st.warning("먼저 3개의 재료를 선택해주세요.")
