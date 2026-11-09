import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
# 设置中文字体与负号正常显示，适配常见操作系统
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Songti SC', 'Heiti SC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理参数与运行常量
# ==========================================
DT = 1.0 / 6.0              # 10分钟离散步长 (h)
STEPS_PER_DAY = 144         # 每天144个时步
BATTERY_CAP = 12000.0       # 电池容量 (kWh)
P_MAX = 5000.0              # PCS最大充放电功率 (kW)
E_PCS_MAX = P_MAX * DT      # 单步最大充放电量: 833.3333 kWh
ETA_CH = 0.90               # 充电效率
ETA_DIS = 0.90              # 放电效率
S_MIN = 1200.0              # 安全储电下限 (kWh)
S_MAX = 10800.0             # 安全储电上限 (kWh)
S_INIT_TOTAL = 6000.0       # 2025-01-01 0:00 初始电量 (kWh)
GAMMA_END = 0.01            # 日末储电量残值激励系数 (元/kWh)

# 表1 指定采样的6个10分钟时段及对应时步索引 (10:00, 12:00, 14:00, 16:00, 18:00, 20:00 后10分钟)
TABLE1_KEYS = ['10:00-10:10', '12:00-12:10', '14:00-14:10', '16:00-16:10', '18:00-18:10', '20:00-20:10']
TABLE1_INDICES = [60, 72, 84, 96, 108, 120]

# 表2 指定充放电的6个4小时时段区间
TABLE2_BLOCKS = [
    ('0:00-4:00',   0, 24),
    ('4:00-8:00',  24, 48),
    ('8:00-12:00', 48, 72),
    ('12:00-16:00', 72, 96),
    ('16:00-20:00', 96, 120),
    ('20:00-24:00', 120, 144)
]

TYPICAL_DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']

# ==========================================
# 2. 数据读取与预报单调保形插值处理
# ==========================================
def load_data():
    """直接读取真实的附件2、附件3、附件4数据文件，完成时间对齐与预报插值展开"""
    print(">>> 正在加载 附件2.xlsx、附件3.xlsx、附件4.xlsx 原始数据...")
    df_load_raw = pd.read_excel('附件2.xlsx', sheet_name='小区负载')
    df_pv_raw = pd.read_excel('附件2.xlsx', sheet_name='光伏发电实际功率')
    df_price_raw = pd.read_excel('附件4.xlsx', sheet_name='Sheet1')
    df_fc_raw = pd.read_excel('附件3.xlsx', sheet_name='Sheet1')

    # 规范化日期格式为 YYYY-MM-DD
    df_load_raw['date_str'] = pd.to_datetime(df_load_raw.iloc[:, 0]).dt.strftime('%Y-%m-%d')
    df_pv_raw['date_str'] = pd.to_datetime(df_pv_raw.iloc[:, 0]).dt.strftime('%Y-%m-%d')
    df_price_raw['date_str'] = pd.to_datetime(df_price_raw.iloc[:, 0]).dt.strftime('%Y-%m-%d')
    df_fc_raw['date_str'] = pd.to_datetime(df_fc_raw['日期']).dt.strftime('%Y-%m-%d')

    date_list = sorted(list(df_load_raw['date_str'].unique()))
    
    P_load_dict, P_pv_dict, Price_dict = {}, {}, {}
    for d in date_list:
        P_load_dict[d] = df_load_raw[df_load_raw['date_str'] == d].iloc[0, 1:145].values.astype(float)
        P_pv_dict[d] = df_pv_raw[df_pv_raw['date_str'] == d].iloc[0, 1:145].values.astype(float)
        Price_dict[d] = df_price_raw[df_price_raw['date_str'] == d].iloc[0, 1:145].values.astype(float)

    # 使用 PCHIP 单调保形插值将 24 小时整点预报平滑展开至 144 个 10 分钟离散步
    PV_forecast_dict = {}
    for d in date_list:
        PV_forecast_dict[d] = {}
        sub_fc = df_fc_raw[df_fc_raw['date_str'] == d]
        for _, row in sub_fc.iterrows():
            t_str = str(row['预报时刻']).strip()
            y_24h = [float(row[f'预报{h}小时']) for h in range(1, 25)]
            x_pts = np.arange(1, 25)
            # 引入起点 0:00 (功率为0) 形成平滑连续边界
            x_pts = np.insert(x_pts, 0, 0)
            y_24h = np.insert(y_24h, 0, 0.0)
            pchip = PchipInterpolator(x_pts, y_24h, extrapolate=True)
            x_interp = np.linspace(0, 24, 144)
            y_interp = np.maximum(0.0, pchip(x_interp))
            PV_forecast_dict[d][t_str] = y_interp

    return date_list, P_load_dict, P_pv_dict, Price_dict, PV_forecast_dict

# ==========================================
# 3. 线性规划核心求解器 (日前与日内MPC)
# ==========================================
def solve_day_ahead_lp(c_price, E_load_fc, E_pv_fc, S_init, S_min_arr):
    """日前规划层LP：求解基准计划购电量与计划充放电策略"""
    N = STEPS_PER_DAY
    c_obj = np.zeros(4 * N)
    c_obj[0:N] = c_price
    c_obj[4 * N - 1] = -GAMMA_END

    # 能量平衡约束: -E_grid + E_ch - E_dis <= E_pv - E_load
    A_ub = np.zeros((N, 4 * N))
    b_ub = np.zeros(N)
    for t in range(N):
        A_ub[t, t] = -1.0
        A_ub[t, N + t] = 1.0
        A_ub[t, 2 * N + t] = -1.0
        b_ub[t] = E_pv_fc[t] - E_load_fc[t]

    # 储能状态方程: S_t - S_{t-1} - eta_ch*E_ch,t + (1/eta_dis)*E_dis,t = 0
    A_eq = np.zeros((N, 4 * N))
    b_eq = np.zeros(N)
    for t in range(N):
        A_eq[t, 3 * N + t] = 1.0
        A_eq[t, N + t] = -ETA_CH
        A_eq[t, 2 * N + t] = 1.0 / ETA_DIS
        if t == 0:
            b_eq[t] = S_init
        else:
            A_eq[t, 3 * N + t - 1] = -1.0
            b_eq[t] = 0.0

    bounds = []
    for _ in range(N): bounds.append((0.0, None))
    for _ in range(N): bounds.append((0.0, E_PCS_MAX))
    for _ in range(N): bounds.append((0.0, E_PCS_MAX))
    for t in range(N): bounds.append((S_min_arr[t], S_MAX))

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        return res.x[0:N]
    return np.maximum(0.0, E_load_fc - E_pv_fc)

def solve_intraday_mpc_lp(tau, c_price_rem, E_plan_rem, E_load_fc_rem, E_pv_fc_rem, S_real_tau):
    """日内滚动MPC层LP：在非对称惩罚(1.5x增购 / -0.5x核减返还)下滚动优化剩余时段购电量"""
    K = len(c_price_rem)
    c_obj = np.zeros(5 * K)
    c_obj[0:K] = 1.5 * c_price_rem
    c_obj[K:2 * K] = -0.5 * c_price_rem
    c_obj[5 * K - 1] = -GAMMA_END

    A_ub = np.zeros((2 * K, 5 * K))
    b_ub = np.zeros(2 * K)
    for k in range(K):
        A_ub[k, k] = -1.0
        A_ub[k, K + k] = 1.0
        A_ub[k, 2 * K + k] = 1.0
        A_ub[k, 3 * K + k] = -1.0
        b_ub[k] = E_plan_rem[k] + E_pv_fc_rem[k] - E_load_fc_rem[k]

        A_ub[K + k, k] = -1.0
        A_ub[K + k, K + k] = 1.0
        b_ub[K + k] = E_plan_rem[k]

    A_eq = np.zeros((K, 5 * K))
    b_eq = np.zeros(K)
    for k in range(K):
        A_eq[k, 4 * K + k] = 1.0
        A_eq[k, 2 * K + k] = -ETA_CH
        A_eq[k, 3 * K + k] = 1.0 / ETA_DIS
        if k == 0:
            b_eq[k] = S_real_tau
        else:
            A_eq[k, 4 * K + k - 1] = -1.0
            b_eq[k] = 0.0

    bounds = []
    for _ in range(K): bounds.append((0.0, None))
    for _ in range(K): bounds.append((0.0, None))
    for _ in range(K): bounds.append((0.0, E_PCS_MAX))
    for _ in range(K): bounds.append((0.0, E_PCS_MAX))
    for _ in range(K): bounds.append((S_MIN, S_MAX))

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        d_plus = res.x[0:K]
        d_minus = res.x[K:2 * K]
        return E_plan_rem + d_plus - d_minus
    return E_plan_rem

# ==========================================
# 4. 日内 EMS 物理执行仿真状态机
# ==========================================
def simulate_realtime_ems(E_grid_exec, E_load_real, E_pv_real, S_prev):
    """实时闭环跟踪：储能响应不平衡量，不足部分触发 5 倍瞬时动态电价紧急购电"""
    N = len(E_grid_exec)
    E_ch_real = np.zeros(N)
    E_dis_real = np.zeros(N)
    E_em_real = np.zeros(N)
    S_real = np.zeros(N)

    s_curr = S_prev
    for t in range(N):
        gap = E_load_real[t] - (E_grid_exec[t] + E_pv_real[t])
        if gap <= 0.0:
            surplus = -gap
            max_ch = min(E_PCS_MAX, (S_MAX - s_curr) / ETA_CH)
            e_ch = min(surplus, max_ch)
            e_dis = 0.0
            e_em = 0.0
            s_curr = s_curr + ETA_CH * e_ch
        else:
            deficit = gap
            max_dis = min(E_PCS_MAX, (s_curr - S_MIN) * ETA_DIS)
            e_dis = min(deficit, max_dis)
            e_ch = 0.0
            e_em = deficit - e_dis
            s_curr = s_curr - e_dis / ETA_DIS

        E_ch_real[t] = e_ch
        E_dis_real[t] = e_dis
        E_em_real[t] = e_em
        S_real[t] = s_curr

    return E_ch_real, E_dis_real, E_em_real, S_real

# ==========================================
# 5. 全年推演与求解流水线
# ==========================================
def run_all_models(date_list, P_load_dict, P_pv_dict, Price_dict, PV_forecast_dict):
    results_4_2 = {}
    results_4_3 = {}
    S_curr_42 = S_INIT_TOTAL
    S_curr_43 = S_INIT_TOTAL

    print("\n>>> 开始全周期（2025.1.1 - 2025.12.31）长时序高保真推演...")
    for idx, d in enumerate(date_list):
        P_load_real = P_load_dict[d]
        P_pv_real = P_pv_dict[d]
        c_price = Price_dict[d]
        E_load_real = P_load_real * DT
        E_pv_real = P_pv_real * DT

        # 7天滑动历史均值基准预测
        if idx >= 7:
            load_hist = [P_load_dict[date_list[idx - k]] for k in range(1, 8)]
            pv_hist = [P_pv_dict[date_list[idx - k]] for k in range(1, 8)]
            P_load_base = np.mean(load_hist, axis=0)
            P_pv_base = np.mean(pv_hist, axis=0)
        else:
            P_load_base = P_load_real.copy()
            P_pv_base = P_pv_real.copy()

        # ---------- 求解问题 4-2 (日前鲁棒分位数防御) ----------
        E_load_fc_42 = (1.03 * P_load_base) * DT
        E_pv_fc_42 = (0.95 * P_pv_base) * DT
        S_min_arr_42 = np.full(STEPS_PER_DAY, S_MIN)
        S_min_arr_42[36:] = S_MIN + 300.0  # 6:00 后施加安全防御死区

        plan_42 = solve_day_ahead_lp(c_price, E_load_fc_42, E_pv_fc_42, S_curr_42, S_min_arr_42)
        ch_42, dis_42, em_42, S_42 = simulate_realtime_ems(plan_42, E_load_real, E_pv_real, S_curr_42)
        s0_42 = S_curr_42
        S_curr_42 = S_42[-1]

        cost_plan_42 = np.sum(c_price * plan_42)
        cost_em_42 = np.sum(5.0 * c_price * em_42)
        cost_total_42 = cost_plan_42 + cost_em_42

        results_4_2[d] = {
            'plan': plan_42, 'ch': ch_42, 'dis': dis_42, 'em': em_42,
            'S': S_42, 's0': s0_42, 's24': S_42[-1], 'cost': cost_total_42,
            'plan_vol': np.sum(plan_42), 'plan_cost': cost_plan_42
        }

        # ---------- 求解问题 4-3 (日内多时序滚动MPC纠偏) ----------
        pv_fc_00 = PV_forecast_dict[d].get('0:00', P_pv_base) * DT
        E_load_fc_43 = P_load_base * DT
        S_min_arr_43 = np.full(STEPS_PER_DAY, S_MIN)

        plan_43 = solve_day_ahead_lp(c_price, E_load_fc_43, pv_fc_00, S_curr_43, S_min_arr_43)
        exec_43 = plan_43.copy()

        s_mpc_curr = S_curr_43
        s0_43 = S_curr_43
        tau_cuts = [(0, 36, '0:00'), (36, 72, '6:00'), (72, 108, '12:00'), (108, 144, '18:00')]
        
        ch_43_full = np.zeros(STEPS_PER_DAY)
        dis_43_full = np.zeros(STEPS_PER_DAY)
        em_43_full = np.zeros(STEPS_PER_DAY)
        S_43_full = np.zeros(STEPS_PER_DAY)

        for t_st, t_ed, cast_k in tau_cuts:
            if t_st > 0:
                pv_rem = PV_forecast_dict[d].get(cast_k, P_pv_base)[t_st:] * DT
                load_rem = E_load_fc_43[t_st:]
                plan_rem = plan_43[t_st:]
                price_rem = c_price[t_st:]
                exec_43[t_st:] = solve_intraday_mpc_lp(t_st, price_rem, plan_rem, load_rem, pv_rem, s_mpc_curr)

            c_seg, d_seg, e_seg, s_seg = simulate_realtime_ems(
                exec_43[t_st:t_ed], E_load_real[t_st:t_ed], E_pv_real[t_st:t_ed], s_mpc_curr
            )
            ch_43_full[t_st:t_ed] = c_seg
            dis_43_full[t_st:t_ed] = d_seg
            em_43_full[t_st:t_ed] = e_seg
            S_43_full[t_st:t_ed] = s_seg
            s_mpc_curr = s_seg[-1]

        S_curr_43 = S_43_full[-1]

        d_plus = np.maximum(0.0, exec_43 - plan_43)
        d_minus = np.maximum(0.0, plan_43 - exec_43)
        cost_plan_base = np.sum(c_price * plan_43)
        cost_adj_net = np.sum(1.5 * c_price * d_plus - 0.5 * c_price * d_minus)
        cost_em_43 = np.sum(5.0 * c_price * em_43_full)
        cost_total_43 = cost_plan_base + cost_adj_net + cost_em_43

        results_4_3[d] = {
            'plan': plan_43, 'adj': exec_43, 'ch': ch_43_full, 'dis': dis_43_full,
            'em': em_43_full, 'S': S_43_full, 's0': s0_43, 's24': S_43_full[-1],
            'cost': cost_total_43, 'exec_vol': np.sum(exec_43), 'plan_vol': np.sum(plan_43),
            'plan_cost': cost_plan_base
        }

    print(">>> 全年推演完成！")
    return results_4_2, results_4_3

# ==========================================
# 6. 控制台格式化输出：表1、表2、表3 深度美化展示
# ==========================================
def display_table1(results, model_label, is_q3=False):
    print(f"\n{'='*98}")
    print(f"       【{model_label}】表1  微网在指定时间段的购电量及全天的购电量和购电费")
    print(f"{'='*98}")
    
    for d in TYPICAL_DATES:
        res = results[d]
        p_vec = res['adj'] if is_q3 else res['plan']
        vol_total = np.sum(p_vec)
        cost_total = res['cost']

        print(f"\n▶ 典型日期: {d}")
        print(f"+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+")
        print(f"| {'时间段':<16} | {'购电量(kWh)':>10} | {'时间段':<16} | {'购电量(kWh)':>10} | {'时间段':<16} | {'购电量(kWh)':>10} |")
        print(f"+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+")
        print(f"| {'10:00-10:10':<16} | {p_vec[60]:>10.4f} | {'12:00-12:10':<16} | {p_vec[72]:>10.4f} | {'14:00-14:10':<16} | {p_vec[84]:>10.4f} |")
        print(f"| {'16:00-16:10':<16} | {p_vec[96]:>10.4f} | {'18:00-18:10':<16} | {p_vec[108]:>10.4f} | {'20:00-20:10':<16} | {p_vec[120]:>10.4f} |")
        print(f"+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+")
        print(f"| {'全天购电量(kWh)':<16} | {vol_total:>10.4f} | {'全天购电费(元)':<16} | {cost_total:>10.4f} | {'-':<16} | {'-':>10} |")
        print(f"+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+{'-'*18}+{'-'*12}+")

    print(f"\n>>> 【{model_label}】表1 4个指定日期横向汇总一览表:")
    print(f"+{'-'*12}+" + "+".join([f"{'-'*13}" for _ in range(6)]) + f"+{'-'*15}+{'-'*14}+")
    headers = ["日期"] + TABLE1_KEYS + ["全天购电量(kWh)", "全天购电费(元)"]
    print(f"| {headers[0]:<10} | " + " | ".join([f"{h:>11}" for h in headers[1:7]]) + f" | {headers[7]:>13} | {headers[8]:>12} |")
    print(f"+{'-'*12}+" + "+".join([f"{'-'*13}" for _ in range(6)]) + f"+{'-'*15}+{'-'*14}+")
    for d in TYPICAL_DATES:
        res = results[d]
        p_vec = res['adj'] if is_q3 else res['plan']
        vals = [f"{p_vec[idx]:.4f}" for idx in TABLE1_INDICES]
        print(f"| {d:<10} | " + " | ".join([f"{v:>11}" for v in vals]) + f" | {np.sum(p_vec):>13.4f} | {res['cost']:>12.4f} |")
    print(f"+{'-'*12}+" + "+".join([f"{'-'*13}" for _ in range(6)]) + f"+{'-'*15}+{'-'*14}+")

def display_table2(results, model_label):
    print(f"\n{'='*98}")
    print(f"  【{model_label}】表2  储能设备在指定时间段的充放电量及0:00和24:00的储电量 (单位: kWh)")
    print(f"{'='*98}")
    
    for d in TYPICAL_DATES:
        res = results[d]
        print(f"\n▶ 典型日期: {d}")
        print(f"+{'-'*14}+{'-'*13}+{'-'*13}+{'-'*14}+{'-'*13}+{'-'*13}+")
        print(f"| {'时间段':<12} | {'充电量(kWh)':>11} | {'放电量(kWh)':>11} | {'时间段':<12} | {'充电量(kWh)':>11} | {'放电量(kWh)':>11} |")
        print(f"+{'-'*14}+{'-'*13}+{'-'*13}+{'-'*14}+{'-'*13}+{'-'*13}+")
        
        ch1 = np.sum(res['ch'][0:24]); dis1 = np.sum(res['dis'][0:24])
        ch2 = np.sum(res['ch'][24:48]); dis2 = np.sum(res['dis'][24:48])
        print(f"| {'0:00-4:00':<12} | {ch1:>11.4f} | {dis1:>11.4f} | {'4:00-8:00':<12} | {ch2:>11.4f} | {dis2:>11.4f} |")

        ch3 = np.sum(res['ch'][48:72]); dis3 = np.sum(res['dis'][48:72])
        ch4 = np.sum(res['ch'][72:96]); dis4 = np.sum(res['dis'][72:96])
        print(f"| {'8:00-12:00':<12} | {ch3:>11.4f} | {dis3:>11.4f} | {'12:00-16:00':<12} | {ch4:>11.4f} | {dis4:>11.4f} |")

        ch5 = np.sum(res['ch'][96:120]); dis5 = np.sum(res['dis'][96:120])
        ch6 = np.sum(res['ch'][120:144]); dis6 = np.sum(res['dis'][120:144])
        print(f"| {'16:00-20:00':<12} | {ch5:>11.4f} | {dis5:>11.4f} | {'20:00-24:00':<12} | {ch6:>11.4f} | {dis6:>11.4f} |")
        print(f"+{'-'*14}+{'-'*13}+{'-'*13}+{'-'*14}+{'-'*13}+{'-'*13}+")
        print(f"| {'0:00储电量':<12} | {res['s0']:>11.4f} | {'-':>11} | {'24:00储电量':<12} | {res['s24']:>11.4f} | {'-':>11} |")
        print(f"+{'-'*14}+{'-'*13}+{'-'*13}+{'-'*14}+{'-'*13}+{'-'*13}+")

def display_table3(results, model_label):
    print(f"\n{'='*108}")
    print(f"                      【{model_label}】表3  微网在指定日期的紧急购电量")
    print(f"{'='*108}")
    
    day_records = {}
    max_len = 0
    for d in TYPICAL_DATES:
        em_vec = results[d]['em']
        em_idx = np.where(em_vec > 0.01)[0]
        records = []
        if len(em_idx) > 0:
            curr_s = em_idx[0]; curr_p = em_idx[0]
            for i in em_idx[1:]:
                if i == curr_p + 1:
                    curr_p = i
                else:
                    h1, m1 = divmod(curr_s * 10, 60); h2, m2 = divmod((curr_p + 1) * 10, 60)
                    t_str = f"{h1:02d}:{m1:02d}-24:00" if h2 == 24 and m2 == 0 else f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
                    records.append((t_str, np.sum(em_vec[curr_s:curr_p+1])))
                    curr_s = i; curr_p = i
            h1, m1 = divmod(curr_s * 10, 60); h2, m2 = divmod((curr_p + 1) * 10, 60)
            t_str = f"{h1:02d}:{m1:02d}-24:00" if h2 == 24 and m2 == 0 else f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
            records.append((t_str, np.sum(em_vec[curr_s:curr_p+1])))
        day_records[d] = records
        if len(records) > max_len:
            max_len = len(records)

    print(f"+{'-'*25}+{'-'*25}+{'-'*25}+{'-'*25}+")
    print(f"| {TYPICAL_DATES[0]:^23} | {TYPICAL_DATES[1]:^23} | {TYPICAL_DATES[2]:^23} | {TYPICAL_DATES[3]:^23} |")
    print(f"+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+")
    print(f"| {'时间段':^11} | {'购电量':^9} | {'时间段':^11} | {'购电量':^9} | {'时间段':^11} | {'购电量':^9} | {'时间段':^11} | {'购电量':^9} |")
    print(f"+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+")

    if max_len == 0:
        print(f"| {'全天无缺额':^11} | {0.0:>9.4f} | {'全天无缺额':^11} | {0.0:>9.4f} | {'全天无缺额':^11} | {0.0:>9.4f} | {'全天无缺额':^11} | {0.0:>9.4f} |")
    else:
        for r_idx in range(max_len):
            row_str = "|"
            for d in TYPICAL_DATES:
                recs = day_records[d]
                if r_idx < len(recs):
                    t_str, vol_val = recs[r_idx]
                    row_str += f" {t_str:^11} | {vol_val:>9.4f} |"
                else:
                    if len(recs) == 0 and r_idx == 0:
                        row_str += f" {'全天无缺额':^11} | {0.0:>9.4f} |"
                    else:
                        row_str += f" {'-':^11} | {'-':^9} |"
            print(row_str)
    print(f"+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+{'-'*13}+{'-'*11}+")

# ==========================================
# 7. 导出至 result4-2.xlsx 与 result4-3.xlsx
# ==========================================
def merge_continuous_emergency(em_vec, date_str):
    """
    识别紧急购电向量中的连续激活时步，将其合并为一个连续时间段区间，并累加购电量
    示例：若 78~80 连续有时步缺额，合并为 '13:00-13:30'，购电量取该区间之和
    """
    rows_em = []
    em_idx = np.where(em_vec > 0.01)[0]
    if len(em_idx) == 0:
        return rows_em

    curr_s = em_idx[0]
    curr_p = em_idx[0]
    ranges = []
    for i in em_idx[1:]:
        if i == curr_p + 1:
            curr_p = i
        else:
            ranges.append((curr_s, curr_p))
            curr_s = i
            curr_p = i
    ranges.append((curr_s, curr_p))

    for st, ed in ranges:
        h1, m1 = divmod(st * 10, 60)
        h2, m2 = divmod((ed + 1) * 10, 60)
        t_str = f"{h1:02d}:{m1:02d}-24:00" if h2 == 24 and m2 == 0 else f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
        em_val = np.sum(em_vec[st:ed+1])
        rows_em.append({'日期': date_str, '购电时间段': t_str, '购电量': round(em_val, 4)})

    return rows_em

def export_to_excel(date_list, results_4_2, results_4_3):
    print("\n>>> 正在将完整购电策略写入 Excel 结果文件...")
    target_dates = [d for d in date_list if d >= '2025-02-01']

    # 尝试加载模板列名结构，或自动构建标准 144 步时间段列名
    interval_cols = []
    template_found = False
    if os.path.exists('result4-2.xlsx'):
        try:
            df_tpl = pd.read_excel('result4-2.xlsx', sheet_name='计划购电量', nrows=1)
            if len(df_tpl.columns) == 147:
                interval_cols = df_tpl.columns[1:145].tolist()
                template_found = True
        except Exception:
            pass

    if not template_found:
        for t in range(STEPS_PER_DAY):
            st_m = t * 10; ed_m = (t + 1) * 10
            h1, m1 = divmod(st_m, 60); h2, m2 = divmod(ed_m, 60)
            if h2 == 24 and m2 == 0:
                interval_cols.append("0:00-0:10+1")
            else:
                interval_cols.append(f"{h1}:{m1:02d}-{h2}:{m2:02d}")

    # ---------------- 写入 result4-2.xlsx ----------------
    with pd.ExcelWriter('result4-2.xlsx', engine='openpyxl') as writer:
        rows_plan = []
        for d in target_dates:
            res = results_4_2[d]
            row = [d] + list(np.round(res['plan'], 4)) + [round(res['plan_vol'], 4), round(res['plan_cost'], 4)]
            rows_plan.append(row)
        pd.DataFrame(rows_plan, columns=['日期\\时间'] + interval_cols + ['全天购电量', '全天购电费']
                    ).to_excel(writer, sheet_name='计划购电量', index=False)

        rows_soc = []
        for d in target_dates:
            res = results_4_2[d]
            for b_idx, (b_name, st, ed) in enumerate(TABLE2_BLOCKS):
                ch_val = np.sum(res['ch'][st:ed])
                dis_val = np.sum(res['dis'][st:ed])
                if b_idx == 0:
                    t_mark, s_val = '00:00:00', round(res['s0'], 4)
                elif b_idx == 1:
                    t_mark, s_val = '24:00:00', round(res['s24'], 4)
                else:
                    t_mark, s_val = np.nan, np.nan
                rows_soc.append({
                    '日期': d, '时间段': b_name, '充电量': round(ch_val, 4),
                    '放电量': round(dis_val, 4), '时刻': t_mark, '储电量': s_val
                })
        pd.DataFrame(rows_soc).to_excel(writer, sheet_name='充放电量', index=False)

        # 紧急购电量：连续时间段合并
        rows_em = []
        for d in target_dates:
            rows_em.extend(merge_continuous_emergency(results_4_2[d]['em'], d))
        if len(rows_em) == 0:
            rows_em.append({'日期': np.nan, '购电时间段': np.nan, '购电量': np.nan})
        pd.DataFrame(rows_em).to_excel(writer, sheet_name='紧急购电量', index=False)

    print(" -> [成功保存] result4-2.xlsx (紧急购电连续时间段已合并)")

    # ---------------- 写入 result4-3.xlsx ----------------
    with pd.ExcelWriter('result4-3.xlsx', engine='openpyxl') as writer:
        rows_plan3 = []
        for d in target_dates:
            res = results_4_3[d]
            row = [d] + list(np.round(res['plan'], 4)) + [round(res['plan_vol'], 4), round(res['plan_cost'], 4)]
            rows_plan3.append(row)
        pd.DataFrame(rows_plan3, columns=['日期\\时间'] + interval_cols + ['全天购电量', '全天购电费']
                    ).to_excel(writer, sheet_name='计划购电量', index=False)

        rows_adj = []
        for d in target_dates:
            res = results_4_3[d]
            row = [d] + list(np.round(res['adj'], 4)) + [round(res['exec_vol'], 4), round(res['cost'], 4)]
            rows_adj.append(row)
        pd.DataFrame(rows_adj, columns=['日期\\时间'] + interval_cols + ['全天购电量', '全天购电费']
                    ).to_excel(writer, sheet_name='调整购电量', index=False)

        rows_soc3 = []
        for d in target_dates:
            res = results_4_3[d]
            for b_idx, (b_name, st, ed) in enumerate(TABLE2_BLOCKS):
                ch_val = np.sum(res['ch'][st:ed])
                dis_val = np.sum(res['dis'][st:ed])
                if b_idx == 0:
                    t_mark, s_val = '00:00:00', round(res['s0'], 4)
                elif b_idx == 1:
                    t_mark, s_val = '24:00:00', round(res['s24'], 4)
                else:
                    t_mark, s_val = np.nan, np.nan
                rows_soc3.append({
                    '日期': d, '时间段': b_name, '充电量': round(ch_val, 4),
                    '放电量': round(dis_val, 4), '时刻': t_mark, '储电量': s_val
                })
        pd.DataFrame(rows_soc3).to_excel(writer, sheet_name='充放电量', index=False)

        # 紧急购电量：连续时间段合并
        rows_em3 = []
        for d in target_dates:
            rows_em3.extend(merge_continuous_emergency(results_4_3[d]['em'], d))
        if len(rows_em3) == 0:
            rows_em3.append({'日期': np.nan, '购电时间段': np.nan, '购电量': np.nan})
        pd.DataFrame(rows_em3).to_excel(writer, sheet_name='紧急购电量', index=False)

    print(" -> [成功保存] result4-3.xlsx (紧急购电连续时间段已合并)")

# ==========================================
# 8. 图表生成模块 (图1、图2、图3)
# ==========================================
def load_fixed_price():
    """读取附件1中的固定电价数据，若不存在则按分时峰谷平标准电价构建基准向量"""
    if os.path.exists('附件1.xlsx'):
        try:
            df1 = pd.read_excel('附件1.xlsx')
            for col in df1.columns:
                if '电价' in str(col):
                    vals = df1[col].dropna().values.astype(float)
                    if len(vals) >= STEPS_PER_DAY:
                        return vals[:STEPS_PER_DAY]
        except Exception:
            pass
    # 标准峰谷分时电价基准 (谷:0.25, 平:0.55, 峰:0.85, 尖:0.90)
    p_fixed = np.full(STEPS_PER_DAY, 0.55)
    p_fixed[0:48] = 0.25      # 0:00-8:00 谷时
    p_fixed[48:72] = 0.85     # 8:00-12:00 峰时
    p_fixed[72:102] = 0.55    # 12:00-17:00 平时
    p_fixed[102:126] = 0.90   # 17:00-21:00 尖峰
    p_fixed[126:144] = 0.55   # 21:00-24:00 平时
    return p_fixed

def generate_figure1_power_balance(results_4_3, Price_dict, P_load_dict, P_pv_dict, target_date='2025-06-21'):
    """图1：典型日时变电价与微网功率平衡堆叠图 (1x2上下分栏子图)"""
    res = results_4_3[target_date]
    price = Price_dict[target_date]
    p_load = P_load_dict[target_date]
    p_pv = P_pv_dict[target_date]
    
    p_grid = res['adj'] / DT
    p_dis = res['dis'] / DT
    p_ch = res['ch'] / DT
    p_em = res['em'] / DT
    hours = np.linspace(0, 24, STEPS_PER_DAY)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # 子图1：外部电网波动电价时序曲线
    ax1.plot(hours, price, color='#E63946', linewidth=2.2, label='实时波动电价 (元/kWh)')
    ax1.axhline(np.mean(price), color='#6C757D', linestyle='--', linewidth=1.2, label=f'日均电价 ({np.mean(price):.2f} 元/kWh)')
    ax1.fill_between(hours, price, alpha=0.15, color='#E63946')
    ax1.set_ylabel('电价 (元/kWh)', fontsize=11, fontweight='bold')
    ax1.set_title(f'(a) {target_date} 外部电网波动电价时序特性', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # 子图2：源-网-荷-储实时功率平衡堆叠图
    ax2.fill_between(hours, 0, p_pv, color='#FFB703', alpha=0.85, label='光伏实际出力 ($P_{pv}$)')
    ax2.fill_between(hours, p_pv, p_pv + p_grid, color='#457B9D', alpha=0.85, label='外网执行购电 ($P_{grid}$)')
    ax2.fill_between(hours, p_pv + p_grid, p_pv + p_grid + p_dis, color='#8338EC', alpha=0.85, label='储能放电支撑 ($P_{dis}$)')
    if np.max(p_em) > 0.01:
        ax2.fill_between(hours, p_pv + p_grid + p_dis, p_pv + p_grid + p_dis + p_em, color='#D90429', alpha=0.9, label='紧急购电保供 ($P_{em}$)')
    
    # 负向填充表示储能吸收充电
    ax2.fill_between(hours, 0, -p_ch, color='#2A9D8F', alpha=0.85, label='储能充电吸收 ($-P_{ch}$)')
    # 小区负荷实际需求线
    ax2.plot(hours, p_load, color='#1D3557', linewidth=2.4, linestyle='-', label='小区负荷需求 ($P_{load}$)')
    ax2.axhline(0, color='black', linewidth=0.9)

    ax2.set_xlabel('时间 (h)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('功率 (kW)', fontsize=11, fontweight='bold')
    ax2.set_title(f'(b) {target_date} 源-网-荷-储功率平衡与削峰填谷响应', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.98), ncol=2, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig('figure1_power_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(">>> [图表输出] 成功保存 图1: figure1_power_balance.png")

def generate_figure2_soc_migration(results_4_2, results_4_3, Price_dict, P_load_dict, P_pv_dict, target_date='2025-06-21'):
    """图2：储能 SOC 动态响应迁移对比图 (固定电价 vs 波动电价4-2 vs 波动电价4-3)"""
    p_fixed = load_fixed_price()
    p_load = P_load_dict[target_date]
    p_pv = P_pv_dict[target_date]
    
    # 计算固定电价下的基准响应
    e_load_fc = (1.03 * p_load) * DT
    e_pv_fc = (0.95 * p_pv) * DT
    s_min_arr = np.full(STEPS_PER_DAY, S_MIN)
    s_min_arr[36:] = S_MIN + 300.0
    plan_fixed = solve_day_ahead_lp(p_fixed, e_load_fc, e_pv_fc, S_INIT_TOTAL, s_min_arr)
    _, _, _, s_fixed = simulate_realtime_ems(plan_fixed, p_load * DT, p_pv * DT, S_INIT_TOTAL)

    s_42 = results_4_2[target_date]['S']
    s_43 = results_4_3[target_date]['S']
    hours = np.linspace(0, 24, STEPS_PER_DAY)

    plt.figure(figsize=(12, 6.5))
    plt.plot(hours, s_fixed, color='#6C757D', linestyle='--', linewidth=2.0, label='固定电价基准策略 (原问题2)')
    plt.plot(hours, s_42, color='#E76F51', linestyle='-.', linewidth=2.2, label='波动电价 Model 4-2 (日前鲁棒策略)')
    plt.plot(hours, s_43, color='#2A9D8F', linestyle='-', linewidth=2.6, label='波动电价 Model 4-3 (日内滚动MPC策略)')

    # 安全界限与警戒线
    plt.axhline(S_MAX, color='#D90429', linestyle=':', linewidth=1.4, label=f'储能安全上限 $S_{{max}}$ ({int(S_MAX)} kWh)')
    plt.axhline(S_MIN, color='#D90429', linestyle=':', linewidth=1.4, label=f'储能安全下限 $S_{{min}}$ ({int(S_MIN)} kWh)')
    plt.fill_between(hours, S_MIN, S_MAX, color='#F8F9FA', alpha=0.6, zorder=0)

    plt.title(f'{target_date} 储能 SOC 动态响应迁移对比轨迹', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('时间 (h)', fontsize=11, fontweight='bold')
    plt.ylabel('储电量 (kWh)', fontsize=11, fontweight='bold')
    plt.xlim(0, 24)
    plt.ylim(800, 11500)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', framealpha=0.92, fontsize=10.5)

    plt.tight_layout()
    plt.savefig('figure2_soc_migration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(">>> [图表输出] 成功保存 图2: figure2_soc_migration.png")

def generate_figure3_cost_breakdown(results_4_2, results_4_3, Price_dict, date_list):
    """图3：Model 4-2 与 Model 4-3 综合费用构成对比图 (特征日对比 + 全年累计对比)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={'width_ratios': [2.2, 1]})

    # 1. 左子图：4个典型特征日费用拆解对比
    bar_width = 0.32
    indices = np.arange(len(TYPICAL_DATES))

    # Model 4-2 拆解项
    cost_base_42 = [np.sum(Price_dict[d] * results_4_2[d]['plan']) for d in TYPICAL_DATES]
    cost_em_42 = [np.sum(5.0 * Price_dict[d] * results_4_2[d]['em']) for d in TYPICAL_DATES]

    # Model 4-3 拆解项
    cost_base_43 = [np.sum(Price_dict[d] * results_4_3[d]['plan']) for d in TYPICAL_DATES]
    cost_adj_43 = []
    for d in TYPICAL_DATES:
        dp = np.maximum(0.0, results_4_3[d]['adj'] - results_4_3[d]['plan'])
        dm = np.maximum(0.0, results_4_3[d]['plan'] - results_4_3[d]['adj'])
        cost_adj_43.append(np.sum(1.5 * Price_dict[d] * dp - 0.5 * Price_dict[d] * dm))
    cost_em_43 = [np.sum(5.0 * Price_dict[d] * results_4_3[d]['em']) for d in TYPICAL_DATES]

    # 绘制 4-2 堆叠柱
    p1 = ax1.bar(indices - bar_width/2, cost_base_42, bar_width, label='4-2 基础计划购电费', color='#457B9D', alpha=0.9)
    p2 = ax1.bar(indices - bar_width/2, cost_em_42, bar_width, bottom=cost_base_42, label='4-2 5倍紧急购电罚款', color='#E63946', alpha=0.9)

    # 绘制 4-3 堆叠柱
    p3 = ax1.bar(indices + bar_width/2, cost_base_43, bar_width, label='4-3 基础计划购电费', color='#2A9D8F', alpha=0.9)
    p4 = ax1.bar(indices + bar_width/2, cost_adj_43, bar_width, bottom=cost_base_43, label='4-3 日内调整净费用', color='#F4A261', alpha=0.9)
    bottom_43 = [cost_base_43[i] + cost_adj_43[i] for i in range(len(TYPICAL_DATES))]
    p5 = ax1.bar(indices + bar_width/2, cost_em_43, bar_width, bottom=bottom_43, label='4-3 5倍紧急购电罚款', color='#D90429', hatch='//', alpha=0.9)

    # 标注总额与降本幅度
    for i in range(len(TYPICAL_DATES)):
        tot42 = cost_base_42[i] + cost_em_42[i]
        tot43 = cost_base_43[i] + cost_adj_43[i] + cost_em_43[i]
        diff_pct = (tot42 - tot43) / tot42 * 100 if tot42 > 0 else 0
        ax1.text(i - bar_width/2, tot42 + 250, f'{tot42:.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax1.text(i + bar_width/2, tot43 + 250, f'{tot43:.0f}\n(-{diff_pct:.1f}%)', ha='center', va='bottom', fontsize=8.5, color='#264653', fontweight='bold')

    ax1.set_xticks(indices)
    ax1.set_xticklabels(TYPICAL_DATES, fontsize=10, fontweight='bold')
    ax1.set_ylabel('费用 (元)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 4 个典型特征日综合费用构成拆解', fontsize=12.5, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax1.legend(loc='upper left', ncol=2, fontsize=8.5, framealpha=0.9)

    # 2. 右子图：2025.2.1 - 2025.12.31 全周期累计对比
    target_dates = [d for d in date_list if d >= '2025-02-01']
    tot_cost_42 = sum(results_4_2[d]['cost'] for d in target_dates) / 1e4
    tot_cost_43 = sum(results_4_3[d]['cost'] for d in target_dates) / 1e4
    tot_em_42 = sum(np.sum(5.0 * Price_dict[d] * results_4_2[d]['em']) for d in target_dates) / 1e4
    tot_em_43 = sum(np.sum(5.0 * Price_dict[d] * results_4_3[d]['em']) for d in target_dates) / 1e4

    bars = ax2.bar(['Model 4-2\n(日前鲁棒)', 'Model 4-3\n(滚动MPC)'], [tot_cost_42, tot_cost_43], color=['#E76F51', '#2A9D8F'], width=0.5, alpha=0.9)
    saved_wan = tot_cost_42 - tot_cost_43
    saved_pct = (saved_wan / tot_cost_42) * 100

    ax2.text(0, tot_cost_42 + 2, f'{tot_cost_42:.1f} 万元\n(含紧急购电 {tot_em_42:.1f}万)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(1, tot_cost_43 + 2, f'{tot_cost_43:.1f} 万元\n(节省 {saved_wan:.1f}万 / -{saved_pct:.1f}%)', ha='center', va='bottom', fontsize=9, color='#1D3557', fontweight='bold')

    ax2.set_ylabel('累计总购电费用 (万元)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 全周期 (2~12月) 累计总费用对比', fontsize=12.5, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.savefig('figure3_cost_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(">>> [图表输出] 成功保存 图3: figure3_cost_breakdown.png")

def generate_all_figures(results_4_2, results_4_3, Price_dict, P_load_dict, P_pv_dict, date_list):
    """一键渲染并导出全部三张论文高标准必选图表"""
    print("\n>>> 正在生成论文图表 (图1、图2、图3)...")
    generate_figure1_power_balance(results_4_3, Price_dict, P_load_dict, P_pv_dict, target_date='2025-06-21')
    generate_figure2_soc_migration(results_4_2, results_4_3, Price_dict, P_load_dict, P_pv_dict, target_date='2025-06-21')
    generate_figure3_cost_breakdown(results_4_2, results_4_3, Price_dict, date_list)

# ==========================================
# 9. 主流程执行入口
# ==========================================
if __name__ == '__main__':
    # 1. 载入输入数据并进行 PCHIP 插值
    date_list, P_load_dict, P_pv_dict, Price_dict, PV_forecast_dict = load_data()

    # 2. 全年求解计算
    results_4_2, results_4_3 = run_all_models(
        date_list, P_load_dict, P_pv_dict, Price_dict, PV_forecast_dict
    )

    # 3. 严格输出【波动电价下问题2】的表1、表2、表3
    print("\n" + "#" * 108)
    print("                   【一、波动电价下 - 问题2的全部报表展示 (日前鲁棒策略)】")
    print("#" * 108)
    display_table1(results_4_2, model_label="问题4-2", is_q3=False)
    display_table2(results_4_2, model_label="问题4-2")
    display_table3(results_4_2, model_label="问题4-2")

    # 4. 严格输出【波动电价下问题3】的表1、表2、表3
    print("\n" + "#" * 108)
    print("                 【二、波动电价下 - 问题3的全部报表展示 (日内滚动MPC策略)】")
    print("#" * 108)
    display_table1(results_4_3, model_label="问题4-3", is_q3=True)
    display_table2(results_4_3, model_label="问题4-3")
    display_table3(results_4_3, model_label="问题4-3")

    # 5. 年度总体对比与论文加分指标汇总
    cost_tot_42 = sum(results_4_2[d]['cost'] for d in date_list if d >= '2025-02-01')
    cost_tot_43 = sum(results_4_3[d]['cost'] for d in date_list if d >= '2025-02-01')
    em_tot_42 = sum(np.sum(results_4_2[d]['em']) for d in date_list if d >= '2025-02-01')
    em_tot_43 = sum(np.sum(results_4_3[d]['em']) for d in date_list if d >= '2025-02-01')

    print("\n" + "=" * 92)
    print("          【三、波动电价下 2025.2.1 - 2025.12.31 综合运行指标与降本效益对比】")
    print("=" * 92)
    print(f" 方案名称             | 全年累计总购电费用 (元) | 全年紧急购电量 (kWh) | 紧急购电总时段数")
    print("-" * 92)
    print(f" 问题4-2 (日前鲁棒)    | {cost_tot_42:>22.2f} | {em_tot_42:>20.2f} | {sum(len(np.where(results_4_2[d]['em']>0.01)[0]) for d in date_list if d >= '2025-02-01'):>16}")
    print(f" 问题4-3 (滚动MPC)     | {cost_tot_43:>22.2f} | {em_tot_43:>20.2f} | {sum(len(np.where(results_4_3[d]['em']>0.01)[0]) for d in date_list if d >= '2025-02-01'):>16}")
    savings = cost_tot_42 - cost_tot_43
    ratio = (savings / cost_tot_42) * 100 if cost_tot_42 > 0 else 0.0
    print("-" * 92)
    print(f" ★ 滚动MPC调控增益    : 累计为微网节省购电成本 {savings:,.2f} 元 (综合降本率: {ratio:.2f}%)")
    print("=" * 92)

    # 6. 生成并导出论文三张必选高分辨率图表
    generate_all_figures(results_4_2, results_4_3, Price_dict, P_load_dict, P_pv_dict, date_list)

    # 7. 保存输出标准 Excel 文件
    export_to_excel(date_list, results_4_2, results_4_3)
    print("\n>>> 全部计算、报表渲染与文件导出已圆满执行完成！")