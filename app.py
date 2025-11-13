import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 한글 폰트 설정 ---
try:
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((f for f in font_path if 'NanumGothic' in f), None)
    malgun_gothic = next((f for f in font_path if 'Malgun' in f), None)

    if nanum_gothic:
        font_prop = fm.FontProperties(fname=nanum_gothic)
        plt.rc('font', family='NanumGothic')
    elif malgun_gothic:
        font_prop = fm.FontProperties(fname=malgun_gothic)
        plt.rc('font', family='Malgun Gothic')
    else:
        font_prop = fm.FontProperties(size=12)

    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.warning(f"한글 폰트를 로드하는 데 실패했습니다. 영문으로 표시될 수 있습니다. 오류: {e}")
    font_prop = fm.FontProperties(size=12)


# --- 2. 2D 열전달 시뮬레이션 함수 ---
def run_2d_heat_simulation(k, L_x, L_y=0.1, rho=150, cp=1000, T_hot=1000+273.15, T_initial=20+273.15, sim_time=5*60):
    alpha = k / (rho * cp)
    nx, ny = 50, 25
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)
    dt = 0.2 * (1 / (alpha * (1/dx**2 + 1/dy**2)))
    nt = int(sim_time / dt)

    time_points = np.linspace(0, sim_time, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial

    for t_step in range(nt):
        T_old = T.copy()
        T[:, 0] = T_hot
        T[:, -1] = T[:, -2]
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                term1 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dy**2
                term2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dx**2
                T[i, j] = T_old[i, j] + alpha * dt * (term1 + term2)
        temp_history_celsius[t_step] = np.mean(T[:, -1]) - 273.15
    return time_points, temp_history_celsius, T - 273.15

# --- 3. 시나리오(재료) 정의 ---
scenarios = {
    '에어로겔': {'k': 0.02, 'rho': 80},
    '세라믹 섬유': {'k': 0.1, 'rho': 150},
    '내화 벽돌': {'k': 1.0, 'rho': 2000},
}

# --- 4. Streamlit UI 구성 ---
st.title("💻 2D 열전달 시뮬레이션")
st.markdown("""
재료와 두께를 선택하여 2D 평판에서의 열 차폐 성능을 시뮬레이션합니다.
- **외부 조건**: 왼쪽 면 1000°C 고정
- **측정**: 오른쪽 면(내부 표면)의 평균 온도 변화
""")
st.sidebar.header("⚙️ 시뮬레이션 설정")
selected_material_name = st.sidebar.selectbox("1. 단열재 종류 선택", options=list(scenarios.keys()))
thickness_cm = st.sidebar.slider("2. 단열재 두께 (cm)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
thickness_m = thickness_cm / 100.0
material_props = scenarios[selected_material_name]
k = material_props['k']
rho = material_props['rho']

if st.sidebar.button("🚀 시뮬레이션 실행"):
    with st.spinner(f"'{selected_material_name}'(두께: {thickness_cm}cm) 시나리오로 시뮬레이션 중..."):
        time_pts, temp_hist, final_temp_dist = run_2d_heat_simulation(k=k, L_x=thickness_m, rho=rho)
        final_temp = temp_hist[-1]
        st.subheader("📊 시뮬레이션 결과")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="재료", value=selected_material_name)
        with col2:
            st.metric(label="최종 내부 표면 평균 온도", value=f"{final_temp:.2f} °C")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(time_pts / 60, temp_hist, label=f"{selected_material_name} (두께: {thickness_cm}cm)")
        ax1.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
        ax1.set_title(f'내부 표면 온도 변화 (두께: {thickness_cm}cm)', fontproperties=font_prop, fontsize=16)
        ax1.set_xlabel('시간 (분)', fontproperties=font_prop)
        ax1.set_ylabel('평균 온도 (°C)', fontproperties=font_prop)
        ax1.legend(prop=font_prop)
        ax1.grid(True)
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, max(500, np.max(temp_hist) * 1.1))
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        im = ax2.imshow(final_temp_dist, cmap='inferno', aspect='auto', extent=[0, thickness_cm, 0, 10])
        fig2.colorbar(im, ax=ax2, label='온도 (°C)')
        ax2.set_title(f'최종 시간(5분)에서의 2D 온도 분포', fontproperties=font_prop, fontsize=16)
        ax2.set_xlabel('두께 방향 (cm)', fontproperties=font_prop)
        ax2.set_ylabel('높이 방향 (cm)', fontproperties=font_prop)
        st.pyplot(fig2)
else:
    st.info("사이드바에서 설정을 마친 후 '시뮬레이션 실행' 버튼을 눌러주세요.")
