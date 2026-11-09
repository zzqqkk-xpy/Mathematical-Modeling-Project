import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# -------------------------------------------------------------
# 0. 绘图环境与学术图表样式配置
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10.5

# -------------------------------------------------------------
# 1. 物理参数与赛题常数设定
# -------------------------------------------------------------
T = 144                       # 每天离散调度时段数 (10分钟一个时步)
DELTA_T = 10.0 / 60.0         # 步长 (h) = 1/6 h
C_CAP = 12000.0               # 储能额定容量 (kWh)
P_MAX = 5000.0                # 最大充放电功率 (kW)
E_RATE_MAX = P_MAX * DELTA_T  # 单步最大充放电量 (kWh) = 833.3333 kWh
S_MIN = 1200.0                # 储能电量下限 (kWh)
S_MAX = 10800.0               # 储能电量上限 (kWh)
S_INIT = 6000.0               # 2025.1.1 0:00 初始储电量 (kWh)
ETA_CH = 0.90                 # 充电效率
ETA_DIS = 0.90                # 放电效率
PENALTY_RATIO = 5.0           # 紧急购电惩罚倍率 (5倍交易时刻电价)

# 时间标签列表 (0:10 至 24:00 共144个)
TIME_LABELS = [
    f"{h:02d}:{m:02d}" if not (h == 24 and m == 0) else "24:00" 
    for h in range(25) for m in (range(10, 61, 10) if h < 24 else [0]) 
    if not (h == 0 and m == 0)
][:144]

# -------------------------------------------------------------
# 2. 数据直接读取模块 (无生成回退，严格核验文件)
# -------------------------------------------------------------
def load_contest_data():
    """
    严格从本地读取附件1.xlsx与附件2.xlsx。文件缺失时抛出异常。
    """
    if not os.path.exists("附件1.xlsx"):
        raise FileNotFoundError("错误：未检测到 '附件1.xlsx'，请确保文件已放置在当前运行目录下！")
    if not os.path.exists("附件2.xlsx"):
        raise FileNotFoundError("错误：未检测到 '附件2.xlsx'，请确保文件已放置在当前运行目录下！")

    # 1. 读取附件1 (电价数据)
    df_att1 = pd.read_excel("附件1.xlsx")
    price_cols = [c for c in df_att1.columns if "电价" in str(c)]
    if not price_cols:
        raise ValueError("附件1.xlsx 中未找到包含'电价'的列名，请检查表头格式！")
    price_daily = df_att1[price_cols[0]].values[:144].astype(float)

    # 2. 读取附件2 (2025全年实际小区负载与光伏功率)
    date_range = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    excel_file = pd.ExcelFile("附件2.xlsx")
    sheet_names = excel_file.sheet_names

    if len(sheet_names) >= 2:
        df_load = pd.read_excel(excel_file, sheet_name=0, index_col=0)
        df_pv = pd.read_excel(excel_file, sheet_name=1, index_col=0)
    else:
        df_all = pd.read_excel(excel_file, sheet_name=0, index_col=0)
        if len(df_all) >= 730:
            df_load = df_all.iloc[:365]
            df_pv = df_all.iloc[365:730]
        else:
            raise ValueError("附件2.xlsx 数据行数不足以解析负荷与光伏两张序列表！")

    load_real = df_load.iloc[:365, :144].values.astype(float)
    pv_real = df_pv.iloc[:365, :144].values.astype(float)

    return date_range, price_daily, load_real, pv_real

# -------------------------------------------------------------
# 3. 日前滚动预测模型 (Quantile Safety Buffer)
# -------------------------------------------------------------
def predict_day_ahead(load_history, pv_history, target_day_idx, buffer_factor=1.03):
    """
    在每天0:00预测未来24小时负荷与光伏。
    结合报童临界分位数准则注入安全缓冲系数 buffer_factor。
    """
    if target_day_idx == 0:
        pred_load = load_history[0].copy()
        pred_pv = pv_history[0].copy()
    elif target_day_idx < 7:
        pred_load = np.mean(load_history[:target_day_idx], axis=0)
        pred_pv = np.mean(pv_history[:target_day_idx], axis=0)
    else:
        weights = np.array([0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3])
        pred_load = np.average(load_history[target_day_idx-7:target_day_idx], axis=0, weights=weights)
        pred_pv = np.average(pv_history[target_day_idx-7:target_day_idx], axis=0, weights=weights)

    pred_load_buffered = pred_load * buffer_factor
    pred_pv_buffered = pred_pv * (2.0 - buffer_factor)
    return pred_load_buffered, pred_pv_buffered

# -------------------------------------------------------------
# 4. 日前经济调度线性规划求解器 (0:00 决策)
# -------------------------------------------------------------
def solve_day_ahead_plan(price, load_pred, pv_pred, S_initial):
    """
    构建并求解0:00日前计划购电模型 (LP)。
    """
    c_obj = np.zeros(4 * T)
    c_obj[:T] = price
    c_obj[3 * T + T - 1] = -0.01  # 储能末态微小激励，防止过度透支

    bounds = []
    for _ in range(T): bounds.append((0, None))            # E_grid >= 0
    for _ in range(T): bounds.append((0, E_RATE_MAX))      # 0 <= E_ch <= 833.33 kWh
    for _ in range(T): bounds.append((0, E_RATE_MAX))      # 0 <= E_dis <= 833.33 kWh
    for _ in range(T): bounds.append((S_MIN + 300, S_MAX)) # 保留300kWh安全防御死区

    # 不等式约束: 供电量不可低于负荷预测值
    A_ub = np.zeros((T, 4 * T))
    b_ub = np.zeros(T)
    for t in range(T):
        A_ub[t, t] = -1.0
        A_ub[t, T + t] = 1.0
        A_ub[t, 2 * T + t] = -1.0
        b_ub[t] = (pv_pred[t] - load_pred[t]) * DELTA_T

    # 等式约束: 储能状态转移方程
    A_eq = np.zeros((T, 4 * T))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, 3 * T + t] = 1.0
        if t > 0:
            A_eq[t, 3 * T + t - 1] = -1.0
        else:
            b_eq[t] = S_initial
        A_eq[t, T + t] = -ETA_CH
        A_eq[t, 2 * T + t] = 1.0 / ETA_DIS

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        return res.x[:T]
    else:
        return np.maximum(0, (load_pred - pv_pred) * DELTA_T)

# -------------------------------------------------------------
# 5. 日内能量管理系统 (EMS) 动态闭环仿真
# -------------------------------------------------------------
def simulate_realtime_ems(price, E_grid_plan, load_real_day, pv_real_day, S_start):
    S_curr = S_start
    ch_record = np.zeros(T)
    dis_record = np.zeros(T)
    em_record = np.zeros(T)
    soc_record = np.zeros(T + 1)
    soc_record[0] = S_curr

    for t in range(T):
        e_load = load_real_day[t] * DELTA_T
        e_pv = pv_real_day[t] * DELTA_T
        e_plan = E_grid_plan[t]
        net_gap = e_load - (e_plan + e_pv)

        if net_gap <= 0:
            surplus = -net_gap
            max_ch_allow = min(E_RATE_MAX, (S_MAX - S_curr) / ETA_CH)
            actual_ch = min(surplus, max_ch_allow)
            
            ch_record[t] = actual_ch
            dis_record[t] = 0.0
            em_record[t] = 0.0
            S_curr += actual_ch * ETA_CH
        else:
            deficit = net_gap
            max_dis_allow = min(E_RATE_MAX, (S_curr - S_MIN) * ETA_DIS)
            actual_dis = min(deficit, max_dis_allow)
            
            ch_record[t] = 0.0
            dis_record[t] = actual_dis
            S_curr -= actual_dis / ETA_DIS
            
            unmet = deficit - actual_dis
            em_record[t] = unmet if unmet > 1e-4 else 0.0

        soc_record[t + 1] = S_curr

    return ch_record, dis_record, em_record, soc_record

# -------------------------------------------------------------
# 6. 全年滚动主循环
# -------------------------------------------------------------
def run_simulation(date_range, price_daily, load_all, pv_all, buffer_factor=1.03):
    days_total = len(date_range)
    all_grid_plan = np.zeros((days_total, T))
    all_ch = np.zeros((days_total, T))
    all_dis = np.zeros((days_total, T))
    all_em = np.zeros((days_total, T))
    all_soc = np.zeros((days_total, T + 1))
    
    current_S = S_INIT
    for d in range(days_total):
        l_pred, p_pred = predict_day_ahead(load_all, pv_all, d, buffer_factor=buffer_factor)
        grid_plan = solve_day_ahead_plan(price_daily, l_pred, p_pred, current_S)
        ch, dis, em, soc = simulate_realtime_ems(price_daily, grid_plan, load_all[d], pv_all[d], current_S)
        
        all_grid_plan[d] = grid_plan
        all_ch[d] = ch
        all_dis[d] = dis
        all_em[d] = em
        all_soc[d] = soc
        current_S = soc[-1]

    return all_grid_plan, all_ch, all_dis, all_em, all_soc

# -------------------------------------------------------------
# 7. 方案A：灵敏度分析与报童最优分位数权衡计算
# -------------------------------------------------------------
def run_sensitivity_tradeoff(date_range, price_daily, load_all, pv_all):
    """
    多重敏感度推演：
    1. 扫描日前安全储备裕度系数 (Buffer Multiplier)，揭示日前过备弃光与日内欠备受罚的经典U型成本凹槽。
    2. 扫描惩罚倍数 k 在 [1, 10] 范围内的总费用风险暴露。
    """
    print("\n>>> 正在运行方案 A：安全裕度与惩罚倍率敏感度权衡推演...")
    buffer_factors = np.linspace(0.98, 1.08, 6)
    buffer_results = []

    # 选取代表性时段进行快速精准权衡评估 (2月至12月各月第15天)
    sample_day_indices = [
        np.where(date_range == f"2025-{m:02d}-15")[0][0] 
        for m in range(2, 13)
    ]

    for bf in buffer_factors:
        tot_plan_cost = 0.0
        tot_em_energy = 0.0
        tot_em_cost = 0.0
        current_S = S_INIT

        for d_idx in sample_day_indices:
            l_pred, p_pred = predict_day_ahead(load_all, pv_all, d_idx, buffer_factor=bf)
            plan = solve_day_ahead_plan(price_daily, l_pred, p_pred, current_S)
            _, _, em, soc = simulate_realtime_ems(price_daily, plan, load_all[d_idx], pv_all[d_idx], current_S)
            
            tot_plan_cost += np.sum(plan * price_daily)
            tot_em_energy += np.sum(em)
            tot_em_cost += np.sum(em * price_daily * PENALTY_RATIO)
            current_S = soc[-1]

        buffer_results.append({
            "buffer": bf,
            "plan_cost": tot_plan_cost,
            "em_energy": tot_em_energy,
            "em_cost": tot_em_cost,
            "total_cost": tot_plan_cost + tot_em_cost
        })

    df_buffer = pd.DataFrame(buffer_results)

    # 扫描紧急惩罚倍率 k 在基准策略下的影响
    k_values = np.linspace(1, 10, 10)
    k_costs = []
    base_plan_cost = buffer_results[2]["plan_cost"]
    base_em_energy = buffer_results[2]["em_energy"]
    base_base_em_cost = buffer_results[2]["em_cost"] / PENALTY_RATIO

    for k in k_values:
        k_costs.append(base_plan_cost + base_base_em_cost * k)

    return df_buffer, k_values, k_costs

# -------------------------------------------------------------
# 8. 绘制四张符合国赛标准的论文插图 (含方案A)
# -------------------------------------------------------------
def plot_academic_figures(date_range, price_daily, load_all, pv_all, all_grid_plan, all_ch, all_dis, all_em, all_soc):
    time_axes = np.linspace(0, 24, 144)
    target_dates = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    indices = [np.where(date_range == d)[0][0] for d in target_dates]

    # 图 1：代表日微网能量平衡与动态调度时序图
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    plot_cases = [("2025-06-21 (夏至日)", indices[1]), ("2025-12-21 (冬至日)", indices[3])]
    
    for col, (title_name, d_idx) in enumerate(plot_cases):
        ax_a = axs1[0, col]
        ax_a.plot(time_axes, load_all[d_idx], label="实际小区负荷", color="#2c3e50", lw=2)
        ax_a.plot(time_axes, pv_all[d_idx], label="实际光伏出力", color="#f39c12", ls="--", lw=2)
        ax_a.plot(time_axes, all_grid_plan[d_idx] / DELTA_T, label="日前计划购电功率", color="#2980b9", lw=1.8)
        if np.sum(all_em[d_idx]) > 0:
            ax_a.fill_between(time_axes, 0, all_em[d_idx] / DELTA_T, color="#e74c3c", alpha=0.6, label="紧急购电功率")
        ax_a.set_title(f"图1(a{col+1}) {title_name} 供需功率平衡全景图", fontweight="bold")
        ax_a.set_ylabel("功率 (kW)")
        ax_a.grid(True, linestyle=":", alpha=0.6)
        ax_a.legend(loc="upper right", framealpha=0.9, fontsize=9)

        ax_b = axs1[1, col]
        ax_b.bar(time_axes, all_dis[d_idx] / DELTA_T, width=0.14, color="#27ae60", label="储能放电 (正)")
        ax_b.bar(time_axes, -all_ch[d_idx] / DELTA_T, width=0.14, color="#8e44ad", label="储能充电 (负)")
        ax_b.axhline(0, color="black", lw=0.8)
        ax_b.set_title(f"图1(b{col+1}) {title_name} 储能系统充放电调度时序", fontweight="bold")
        ax_b.set_xlabel("时间 (小时)")
        ax_b.set_ylabel("充放电功率 (kW)")
        ax_b.set_xlim(0, 24)
        ax_b.grid(True, linestyle=":", alpha=0.6)
        ax_b.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig("图1_代表日微网能量平衡与动态调度时序图.png")
    plt.close()

    # 图 2：四季代表日储电量轨迹与电价联动图
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors = ["#16a085", "#e67e22", "#2980b9", "#c0392b"]
    labels = ["春分日 (2025.3.20)", "夏至日 (2025.6.21)", "秋分日 (2025.9.23)", "冬至日 (2025.12.21)"]

    for i, idx in enumerate(indices):
        ax2.plot(np.linspace(0, 24, 145), all_soc[idx], label=f"{labels[i]} 储电量", color=colors[i], lw=2.2)

    ax2.axhline(S_MAX, color="red", ls=":", lw=1.5, label=f"安全容量上限 ({int(S_MAX)} kWh)")
    ax2.axhline(S_MIN, color="crimson", ls=":", lw=1.5, label=f"安全容量下限 ({int(S_MIN)} kWh)")
    ax2.axhline(S_INIT, color="gray", ls="-.", lw=1.2, label="标称初始储电量 (6000 kWh)")
    
    ax2_twin = ax2.twinx()
    ax2_twin.step(time_axes, price_daily, where="post", color="#7f8c8d", alpha=0.4, lw=1.5, ls="--", label="分时电价曲线")
    ax2_twin.fill_between(time_axes, 0, price_daily, step="post", color="#bdc3c7", alpha=0.15)
    ax2_twin.set_ylabel("分时电价 (元/kWh)", color="#555")

    ax2.set_title("图2 四季代表日储能设备电量轨迹演进与电价套利联动机制图", fontweight="bold", fontsize=13)
    ax2.set_xlabel("调度时刻 (小时)")
    ax2.set_ylabel("储能设备当前储电量 (kWh)")
    ax2.set_xlim(0, 24)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.9, fontsize=9)
    ax2_twin.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig("图2_四季代表日储电量轨迹与电价联动图.png")
    plt.close()

    # 图 3：全年逐月购电费用结构与紧急购电统计
    df_eval = pd.DataFrame({
        "Date": date_range,
        "Plan_Cost": [np.sum(all_grid_plan[i] * price_daily) for i in range(len(date_range))],
        "Em_Cost": [np.sum(all_em[i] * price_daily * PENALTY_RATIO) for i in range(len(date_range))],
        "Em_Energy": [np.sum(all_em[i]) for i in range(len(date_range))]
    })
    df_eval["Month"] = df_eval["Date"].dt.month
    monthly_stats = df_eval.groupby("Month")[["Plan_Cost", "Em_Cost", "Em_Energy"]].sum()
    months = np.arange(1, 13)

    fig3, ax3 = plt.subplots(figsize=(11, 6))
    width = 0.55
    ax3.bar(months, monthly_stats["Plan_Cost"], width, label="日前计划购电费用 (元)", color="#3498db")
    ax3.bar(months, monthly_stats["Em_Cost"], width, bottom=monthly_stats["Plan_Cost"],
            label="日内紧急购电费用 (5倍惩罚)", color="#e74c3c")

    ax3_twin = ax3.twinx()
    ax3_twin.plot(months, monthly_stats["Em_Energy"], color="#d35400", marker="o", lw=2.2, label="月度累计紧急购电量 (kWh)")
    ax3_twin.set_ylabel("紧急购电量 (kWh)", color="#d35400", fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#d35400")

    ax3.set_title("图3 2025 全年逐月微网购电费用结构与紧急购电电量风险分析", fontweight="bold", fontsize=13)
    ax3.set_xlabel("月份")
    ax3.set_ylabel("购电费用 (元)")
    ax3.set_xticks(months)
    ax3.set_xticklabels([f"{m}月" for m in months])
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    l1, b1 = ax3.get_legend_handles_labels()
    l2, b2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(l1 + l2, b1 + b2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("图3_全年逐月购电费用结构与紧急购电统计.png")
    plt.close()

    # 图 4（方案 A）：灵敏度权衡曲线与报童帕累托前沿图
    df_buffer, k_vals, k_costs = run_sensitivity_tradeoff(date_range, price_daily, load_all, pv_all)

    fig4, axs4 = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 4(a): 安全裕度调节与U型总成本曲线 (报童最优平衡验证)
    ax4_a = axs4[0]
    ax4_a.plot(df_buffer["buffer"], df_buffer["total_cost"], color="#c0392b", marker="s", lw=2.5, label="微网综合总购电费用 (元)")
    ax4_a.plot(df_buffer["buffer"], df_buffer["plan_cost"], color="#2980b9", ls="--", lw=1.8, label="日前计划购电成本 (元)")
    ax4_a.plot(df_buffer["buffer"], df_buffer["em_cost"], color="#e67e22", ls=":", lw=1.8, label="日内紧急购电惩罚费 (元)")
    
    # 标注最低成本拐点
    min_idx = df_buffer["total_cost"].idxmin()
    best_buf = df_buffer.loc[min_idx, "buffer"]
    best_cost = df_buffer.loc[min_idx, "total_cost"]
    ax4_a.scatter(best_buf, best_cost, color="#c0392b", s=100, zorder=5)
    ax4_a.annotate(f"报童最优拐点\n裕度={best_buf:.2f}\n最低费用={best_cost:.1f}元", 
                   xy=(best_buf, best_cost), xytext=(best_buf-0.03, best_cost*1.02),
                   arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5), fontweight="bold")
    
    ax4_a.set_title("图4(a) 日前防御裕度对微网费用构成的帕累托权衡 (报童效应)", fontweight="bold")
    ax4_a.set_xlabel("日前负荷安全缓冲系数 (Buffer Multiplier)")
    ax4_a.set_ylabel("样本总购电费用 (元)")
    ax4_a.grid(True, linestyle=":", alpha=0.6)
    ax4_a.legend(loc="upper right", framealpha=0.9)

    # 4(b): 惩罚倍率 k 在 [1, 10] 范围内的费用敏感度响应
    ax4_b = axs4[1]
    ax4_b.plot(k_vals, k_costs, color="#8e44ad", marker="o", lw=2.2, label="微网购电总成本随惩罚倍数增长轨迹")
    ax4_b.axvline(5.0, color="red", ls="--", lw=1.5, label="赛题基准设定 (k = 5.0 倍)")
    ax4_b.set_title("图4(b) 紧急购电惩罚倍率 k 对微网系统经济性的敏感度响应", fontweight="bold")
    ax4_b.set_xlabel("紧急购电惩罚倍数 k")
    ax4_b.set_ylabel("微网总购电成本 (元)")
    ax4_b.grid(True, linestyle=":", alpha=0.6)
    ax4_b.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("图4_日前安全裕度灵敏度权衡与报童帕累托前沿.png")
    plt.close()

    print(">>> 包含方案 A 在内的四张学术插图已全部生成完毕！")

# -------------------------------------------------------------
# 9. 赛题 Excel 导出规范与指定表格打印
# -------------------------------------------------------------
def export_and_print_tables(date_range, price_daily, all_grid_plan, all_ch, all_dis, all_em, all_soc):
    mask = (date_range >= "2025-02-01") & (date_range <= "2025-12-31")
    eval_dates_str = date_range[mask].strftime("%Y/%m/%d")
    
    sub_plan = all_grid_plan[mask]
    sub_ch = all_ch[mask]
    sub_dis = all_dis[mask]
    sub_em = all_em[mask]
    sub_soc = all_soc[mask]

    # Sheet 1: 计划购电量 (包含144个间隔时段以及全天购电量、购电费)
    df_sheet1 = pd.DataFrame(sub_plan, index=eval_dates_str, columns=TIME_LABELS)
    df_sheet1.index.name = "日期"
    df_sheet1["全天购电量"] = np.sum(sub_plan, axis=1).round(4)
    df_sheet1["全天购电费"] = np.dot(sub_plan, price_daily).round(4)

    # Sheet 2: 充放电量
    periods = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    ch_blocks = [sub_ch[:, i*24:(i+1)*24].sum(axis=1).round(4) for i in range(6)]
    dis_blocks = [sub_dis[:, i*24:(i+1)*24].sum(axis=1).round(4) for i in range(6)]
    
    sheet2_data = {}
    for i, p in enumerate(periods):
        sheet2_data[f"{p} 充电量"] = ch_blocks[i]
        sheet2_data[f"{p} 放电量"] = dis_blocks[i]
    sheet2_data["0:00 储电量"] = sub_soc[:, 0].round(4)
    sheet2_data["24:00 储电量"] = sub_soc[:, -1].round(4)
    df_sheet2 = pd.DataFrame(sheet2_data, index=eval_dates_str)
    df_sheet2.index.name = "日期"

    # Sheet 3: 紧急购电量
    em_rows = []
    for d_idx, d_str in enumerate(eval_dates_str):
        em_day = sub_em[d_idx]
        t = 0
        while t < T:
            if em_day[t] > 1e-4:
                start_t = t
                tot_em = 0.0
                while t < T and em_day[t] > 1e-4:
                    tot_em += em_day[t]
                    t += 1
                end_t = t
                start_str = f"{(start_t*10)//60:02d}:{(start_t*10)%60:02d}"
                end_str = f"{(end_t*10)//60:02d}:{(end_t*10)%60:02d}"
                em_rows.append({
                    "日期": d_str,
                    "紧急购电时间段": f"{start_str}-{end_str}",
                    "紧急购电量": round(tot_em, 4)
                })
            else:
                t += 1

    df_sheet3 = pd.DataFrame(em_rows)
    if df_sheet3.empty:
        df_sheet3 = pd.DataFrame(columns=["日期", "紧急购电时间段", "紧急购电量"])

    with pd.ExcelWriter("result2.xlsx", engine="openpyxl") as writer:
        df_sheet1.to_excel(writer, sheet_name="计划购电量")
        df_sheet2.to_excel(writer, sheet_name="充放电量")
        df_sheet3.to_excel(writer, sheet_name="紧急购电量", index=False)
    
    print(">>> 成果文件已导出至 'result2.xlsx' (已包含全天购电量与全天购电费)。")

def display_specified_date_tables(date_range, price_daily, all_grid_plan, all_ch, all_dis, all_em, all_soc):
    target_dates = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    
    print("\n" + "="*80)
    print("【表1格式】微网在指定日期的计划购电量及全天的购电量和购电费 (单位: kWh / 元)")
    print("="*80)
    target_slots = [
        ("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
        ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)
    ]
    
    col_headers = ["时间段 / 统计指标"] + target_dates
    row_list = []
    for slot_name, slot_idx in target_slots:
        row = [slot_name]
        for d in target_dates:
            d_idx = np.where(date_range == d)[0][0]
            row.append(f"{all_grid_plan[d_idx, slot_idx]:.4f}")
        row_list.append(row)
    
    # 全天计划购电量与费用
    tot_e_row = ["全天计划购电量"]
    tot_c_row = ["全天计划购电费"]
    for d in target_dates:
        d_idx = np.where(date_range == d)[0][0]
        tot_e = np.sum(all_grid_plan[d_idx])
        tot_c = np.sum(all_grid_plan[d_idx] * price_daily)
        tot_e_row.append(f"{tot_e:.4f}")
        tot_c_row.append(f"{tot_c:.4f}")
    row_list.append(tot_e_row)
    row_list.append(tot_c_row)
    
    print(pd.DataFrame(row_list, columns=col_headers).to_markdown(index=False))

    print("\n" + "="*80)
    print("【表2格式】储能设备在指定日期的充放电量及始末储电量统计 (单位: kWh)")
    print("="*80)
    periods = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    for d in target_dates:
        d_idx = np.where(date_range == d)[0][0]
        print(f"\n>>> 日期: {d}")
        table2_records = []
        for p_idx, p_name in enumerate(periods):
            ch_val = np.sum(all_ch[d_idx, p_idx*24:(p_idx+1)*24])
            dis_val = np.sum(all_dis[d_idx, p_idx*24:(p_idx+1)*24])
            table2_records.append({"时间段": p_name, "充电量": f"{ch_val:.4f}", "放电量": f"{dis_val:.4f}"})
        print(pd.DataFrame(table2_records).to_markdown(index=False))
        print(f"0:00 储电量: {all_soc[d_idx, 0]:.4f} kWh  |  24:00 储电量: {all_soc[d_idx, -1]:.4f} kWh")

    print("\n" + "="*80)
    print("【表3格式】微网在指定日期的紧急购电量统计表")
    print("="*80)
    em_summary = []
    for d in target_dates:
        d_idx = np.where(date_range == d)[0][0]
        em_day = all_em[d_idx]
        t = 0
        found = False
        while t < T:
            if em_day[t] > 1e-4:
                found = True
                start_t = t
                tot_val = 0.0
                while t < T and em_day[t] > 1e-4:
                    tot_val += em_day[t]
                    t += 1
                end_t = t
                s_str = f"{(start_t*10)//60:02d}:{(start_t*10)%60:02d}"
                e_str = f"{(end_t*10)//60:02d}:{(end_t*10)%60:02d}"
                em_summary.append({"日期": d, "紧急购电时间段": f"{s_str}-{e_str}", "紧急购电量 (kWh)": f"{tot_val:.4f}"})
            else:
                t += 1
        if not found:
            em_summary.append({"日期": d, "紧急购电时间段": "— (无紧急购电)", "紧急购电量 (kWh)": "0.0000"})
    print(pd.DataFrame(em_summary).to_markdown(index=False))

# -------------------------------------------------------------
# 10. 程序主入口
# -------------------------------------------------------------
if __name__ == "__main__":
    date_range, price_daily, load_all, pv_all = load_contest_data()
    all_grid_plan, all_ch, all_dis, all_em, all_soc = run_simulation(date_range, price_daily, load_all, pv_all)
    export_and_print_tables(date_range, price_daily, all_grid_plan, all_ch, all_dis, all_em, all_soc)
    display_specified_date_tables(date_range, price_daily, all_grid_plan, all_ch, all_dis, all_em, all_soc)
    plot_academic_figures(date_range, price_daily, load_all, pv_all, all_grid_plan, all_ch, all_dis, all_em, all_soc)