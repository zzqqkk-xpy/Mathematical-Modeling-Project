import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import linprog

# -------------------------------------------------------------
# 1. 图表字体与学术出版风格配置
# -------------------------------------------------------------
# 自动寻找系统中文支持字体（如SimHei、Noto Sans CJK、Microsoft YaHei等）
chinese_fonts = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 10.5

# -------------------------------------------------------------
# 2. 读取数据与系统物理参数定义
# -------------------------------------------------------------
file_path = '附件1.xlsx'
df = pd.read_excel(file_path)

T = len(df)                  # 时步数: 144个时间段 (对应24小时)
dt = 1.0 / 6.0               # 时间步长: 10分钟 = 1/6 小时

prices = df['电价'].values              # 单位: 元/kWh
load_p = df['小区负载'].values          # 单位: kW
pv_p = df['光伏发电预测功率'].values    # 单位: kW

load_e = load_p * dt         # 每个时段的小区耗电量 (kWh)
pv_e = pv_p * dt             # 每个时段的光伏发电量 (kWh)

# 储能设备物理边界参数 (见题意附录1)
C_cap = 12000.0              # 储能设备额定容量 (kWh)
P_max = 5000.0               # 储能最大充放电功率 (kW)
E_step_max = P_max * dt      # 单时步最大充放电量 (833.3333 kWh)
S_min = 1200.0               # 荷电下限 (kWh, 10% SOC)
S_max = 10800.0              # 荷电上限 (kWh, 90% SOC)
S_init = 6000.0              # 0:00与24:00规定储电量 (kWh)
eta_ch = 0.9                 # 充电效率 (90%)
eta_dis = 0.9                # 放电效率 (90%)

# -------------------------------------------------------------
# 3. 构建线性规划模型 (Linear Programming via HiGHS)
# -------------------------------------------------------------
# 决策变量向量 x 共 4 * T 个变量，顺序定义如下:
# x[0 : T]       : E_grid[t] (微网向外网购电量, kWh)
# x[T : 2T]     : E_ch[t]   (微网向储能充入电量, kWh)
# x[2T : 3T]    : E_dis[t]  (储能向微网释放电量, kWh)
# x[3T : 4T]    : S[t]      (第t个时段结束时的储电量, kWh)
num_vars = 4 * T

# 目标函数：最小化全天购电总费用 min sum( c_t * E_grid[t] )
c = np.zeros(num_vars)
c[:T] = prices

# 变量上下限边界 Bounds
bounds = []
for t in range(T):
    bounds.append((0, None))            # E_grid >= 0
for t in range(T):
    bounds.append((0, E_step_max))       # 0 <= E_ch <= P_max * dt
for t in range(T):
    bounds.append((0, E_step_max))       # 0 <= E_dis <= P_max * dt
for t in range(T):
    bounds.append((S_min, S_max))        # 1200 <= S[t] <= 10800

# 等式约束 A_eq * x = b_eq
# 约束1：储能动态状态转移方程 S[t] = S[t-1] + eta_ch * E_ch[t] - (1/eta_dis) * E_dis[t]
# 约束2：全天储能电量平衡 S[T-1] = S_init = 6000
A_eq = []
b_eq = []

# t = 0 时刻 (S[-1] 即初始 0:00 电量 6000 kWh)
row0 = np.zeros(num_vars)
row0[3*T + 0] = 1.0                      # S[0]
row0[T + 0] = -eta_ch                    # -eta_ch * E_ch[0]
row0[2*T + 0] = 1.0 / eta_dis            # + (1/eta_dis) * E_dis[0]
A_eq.append(row0)
b_eq.append(S_init)

# t = 1 到 T-1
for t in range(1, T):
    row = np.zeros(num_vars)
    row[3*T + t] = 1.0                   # S[t]
    row[3*T + t - 1] = -1.0              # -S[t-1]
    row[T + t] = -eta_ch                 # -eta_ch * E_ch[t]
    row[2*T + t] = 1.0 / eta_dis         # + (1/eta_dis) * E_dis[t]
    A_eq.append(row)
    b_eq.append(0.0)

# 终态约束：24:00 (即 t=143 结束时) 电量恢复 6000 kWh
row_end = np.zeros(num_vars)
row_end[3*T + T - 1] = 1.0
A_eq.append(row_end)
b_eq.append(S_init)

# 不等式约束 A_ub * x <= b_ub
# 约束3：微网提供的电能不可低于小区负载
# E_grid[t] + pv_e[t] + E_dis[t] - E_ch[t] >= load_e[t]
# 整理为标准型: -E_grid[t] + E_ch[t] - E_dis[t] <= pv_e[t] - load_e[t]
A_ub = []
b_ub = []
for t in range(T):
    row = np.zeros(num_vars)
    row[t] = -1.0                        # -E_grid[t]
    row[T + t] = 1.0                     # +E_ch[t]
    row[2*T + t] = -1.0                  # -E_dis[t]
    A_ub.append(row)
    b_ub.append(pv_e[t] - load_e[t])

# -------------------------------------------------------------
# 4. 执行模型求解
# -------------------------------------------------------------
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if not res.success:
    raise ValueError(f"优化求解失败: {res.message}")

print("================== 优化求解成功 ==================")
print(f"全天最小购电总费用: {res.fun:.4f} 元")

# 提取优化变量
E_grid = res.x[:T]
E_ch = res.x[T:2*T]
E_dis = res.x[2*T:3*T]
S = res.x[3*T:4*T]

# 折算为瞬时功率 (kW)
P_grid = E_grid / dt
P_ch = E_ch / dt
P_dis = E_dis / dt

# -------------------------------------------------------------
# 5. 输出格式化结果（表1 与 表2）
# -------------------------------------------------------------
# 表1：指定时段购电量及全天总指标
table1_spec = [
    ("10:00-10:10", 60),
    ("12:00-12:10", 72),
    ("14:00-14:10", 84),
    ("16:00-16:10", 96),
    ("18:00-18:10", 108),
    ("20:00-20:10", 120)
]

print("\n---------------- 表 1：指定时间段购电量 ----------------")
for label, idx in table1_spec:
    print(f"时段 {label:<11} | 计划购电量: {E_grid[idx]:10.4f} kWh")
total_grid_energy = np.sum(E_grid)
total_grid_cost = res.fun
print(f"全天总购电量: {total_grid_energy:.4f} kWh | 全天购电费: {total_grid_cost:.4f} 元")

# 表2：4小时充放电统计与0点/24点储电量
four_hour_intervals = [
    ("0:00-4:00", 0, 24),
    ("4:00-8:00", 24, 48),
    ("8:00-12:00", 48, 72),
    ("12:00-16:00", 72, 96),
    ("16:00-20:00", 96, 120),
    ("20:00-24:00", 120, 144)
]

print("\n---------------- 表 2：储能设备充放电量统计 ----------------")
for label, st, ed in four_hour_intervals:
    ch_sum = np.sum(E_ch[st:ed])
    dis_sum = np.sum(E_dis[st:ed])
    print(f"时段 {label:<11} | 充电量: {ch_sum:10.4f} kWh | 放电量: {dis_sum:10.4f} kWh")
print(f"0:00 储电量  : {S_init:10.4f} kWh | 24:00 储电量: {S[-1]:10.4f} kWh")

# -------------------------------------------------------------
# 6. 保存赛题要求的 Excel 结果文件 (result1.xlsx)
# -------------------------------------------------------------
with pd.ExcelWriter('result1.xlsx') as writer:
    # Sheet 1: 计划购电量 (10分钟间隔)
    df_grid_export = pd.DataFrame({
        '时间': df['时间'],
        '计划购电量(kWh)': np.round(E_grid, 4)
    })
    df_grid_export.to_excel(writer, sheet_name='计划购电量', index=False)

    # Sheet 2: 充放电量
    storage_rows = []
    for label, st, ed in four_hour_intervals:
        storage_rows.append({
            '时间段': label,
            '充电量(kWh)': np.round(np.sum(E_ch[st:ed]), 4),
            '放电量(kWh)': np.round(np.sum(E_dis[st:ed]), 4)
        })
    # 追加 0:00 和 24:00 电量行
    storage_rows.append({'时间段': '0:00 储电量', '充电量(kWh)': S_init, '放电量(kWh)': np.nan})
    storage_rows.append({'时间段': '24:00 储电量', '充电量(kWh)': np.nan, '放电量(kWh)': np.round(S[-1], 4)})
    df_storage_export = pd.DataFrame(storage_rows)
    df_storage_export.to_excel(writer, sheet_name='充放电量', index=False)

print("\n[OK] 完整购电及储能策略已成功保存至 result1.xlsx 文件！")

# -------------------------------------------------------------
# 7. 论文插图绘制 (Figure 1, Figure 2, Figure 3)
# -------------------------------------------------------------
hours = np.linspace(10/60, 24, T)

# 【图1】微网日前协同调度与多源供需功率平衡全景图
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})

# 子图 (a)：功率供求平衡
ax1a.plot(hours, load_p, label='小区负荷需求 $P_{load}$', color='#2c3e50', linewidth=2.2, zorder=4)
ax1a.plot(hours, pv_p, label='光伏预测出力 $P_{pv}$', color='#f39c12', linewidth=2.0, linestyle='--', zorder=3)
ax1a.plot(hours, P_grid, label='外网计划购电功率 $P_{grid}$', color='#2980b9', linewidth=1.8, linestyle='-.', zorder=2)
ax1a.fill_between(hours, 0, pv_p, color='#f39c12', alpha=0.15, label='光伏出力覆盖区域')
ax1a.set_ylabel('功率 (kW)', fontsize=11, fontweight='bold')
ax1a.set_title('(a) 微网全天供需负荷平衡及各发电/用电单元协同调度', fontsize=12, fontweight='bold', pad=10)
ax1a.grid(True, linestyle=':', alpha=0.6)
ax1a.legend(loc='upper right', frameon=True, fontsize=10, facecolor='white', framealpha=0.9, shadow=True)
ax1a.set_xlim(0, 24)

# 子图 (b)：储能充放电调度时序
ax1b.bar(hours, P_dis, width=0.13, label='储能放电功率 $P_{dis}$ (向微网供电)', color='#27ae60', alpha=0.85, zorder=3)
ax1b.bar(hours, -P_ch, width=0.13, label='储能充电功率 $P_{ch}$ (自微网吸电)', color='#8e44ad', alpha=0.85, zorder=3)
ax1b.axhline(0, color='#333333', linewidth=1.0)
ax1b.set_xlabel('时间 (小时 / h)', fontsize=11, fontweight='bold')
ax1b.set_ylabel('储能功率 (kW)', fontsize=11, fontweight='bold')
ax1b.set_title('(b) 储能系统充放电功率调度时序分布 (正: 放电, 负: 充电)', fontsize=12, fontweight='bold', pad=10)
ax1b.set_xticks(np.arange(0, 25, 2))
ax1b.set_xticklabels([f'{h:02d}:00' for h in np.arange(0, 25, 2)])
ax1b.grid(True, linestyle=':', alpha=0.6)
ax1b.legend(loc='lower right', frameon=True, fontsize=10, facecolor='white', framealpha=0.9, shadow=True)

plt.tight_layout()
fig1.savefig('figure1_power_dispatch.png', dpi=300)
print("[OK] 图1已保存: figure1_power_dispatch.png")

# 【图2】储能设备荷电状态 (SOC/储电量) 与分时电价联动套利机制图
fig2, ax_soc = plt.subplots(figsize=(12, 6))
ax_price = ax_soc.twinx()

soc_curve = [S_init] + list(S)
hours_soc = [0.0] + list(hours)

l1 = ax_soc.plot(hours_soc, soc_curve, color='#c0392b', linewidth=2.5, label='储能设备储电量 (kWh)', zorder=4)
ax_soc.axhline(S_max, color='#e74c3c', linestyle=':', linewidth=1.5, label=f'储电上限 ({int(S_max)} kWh)')
ax_soc.axhline(S_min, color='#e67e22', linestyle=':', linewidth=1.5, label=f'储电下限 ({int(S_min)} kWh)')
ax_soc.axhline(S_init, color='#7f8c8d', linestyle='--', linewidth=1.2, label=f'始末平衡基准 ({int(S_init)} kWh)')

l2 = ax_price.plot(hours, prices, color='#2980b9', linewidth=2.0, linestyle='-', label='外网实时电价 (元/kWh)', zorder=3)
ax_price.fill_between(hours, prices, min(prices), color='#3498db', alpha=0.1)

ax_soc.set_xlabel('时间 (小时 / h)', fontsize=11, fontweight='bold')
ax_soc.set_ylabel('储能设备储电量 (kWh)', fontsize=11, fontweight='bold', color='#c0392b')
ax_price.set_ylabel('分时电价 (元/kWh)', fontsize=11, fontweight='bold', color='#2980b9')
ax_soc.tick_params(axis='y', labelcolor='#c0392b')
ax_price.tick_params(axis='y', labelcolor='#2980b9')
ax_soc.set_xlim(0, 24)
ax_soc.set_xticks(np.arange(0, 25, 2))
ax_soc.set_xticklabels([f'{h:02d}:00' for h in np.arange(0, 25, 2)])
ax_soc.set_ylim(0, 12000)
ax_soc.grid(True, linestyle=':', alpha=0.6)

# 图例汇总
lines = l1 + l2 + [
    plt.Line2D([0], [0], color='#e74c3c', linestyle=':', linewidth=1.5),
    plt.Line2D([0], [0], color='#e67e22', linestyle=':', linewidth=1.5),
    plt.Line2D([0], [0], color='#7f8c8d', linestyle='--', linewidth=1.2)
]
labels = ['储能储电量', '分时电价', f'安全上限 ({int(S_max)} kWh)', f'安全下限 ({int(S_min)} kWh)', f'始末平衡基准 ({int(S_init)} kWh)']
ax_soc.legend(lines, labels, loc='upper left', frameon=True, fontsize=10, facecolor='white', framealpha=0.9, shadow=True)

plt.title('储能设备储电动态轨迹与分时电价经济套利联动机制', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
fig2.savefig('figure2_soc_price_coupling.png', dpi=300)
print("[OK] 图2已保存: figure2_soc_price_coupling.png")

# 【图3】竞赛规定时段与区间充放电量对比统计图
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) 表1 指定时间段购电量对比
labels_p1 = [item[0] for item in table1_spec]
vals_p1 = [E_grid[item[1]] for item in table1_spec]
colors_p1 = ['#3498db' if v > 0 else '#bdc3c7' for v in vals_p1]

bars1 = ax3a.bar(labels_p1, vals_p1, color=colors_p1, width=0.55, edgecolor='#2c3e50', linewidth=1)
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax3a.text(bar.get_x() + bar.get_width()/2., h + 10, f'{h:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax3a.text(bar.get_x() + bar.get_width()/2., 10, '0.0', ha='center', va='bottom', fontsize=10, color='gray')
ax3a.set_title('(a) 表1 指定时间段购电量分布 (单位: kWh)', fontsize=12, fontweight='bold', pad=12)
ax3a.set_ylabel('购电量 (kWh)', fontsize=11, fontweight='bold')
ax3a.set_ylim(0, max(vals_p1) * 1.18)
ax3a.grid(True, axis='y', linestyle=':', alpha=0.6)
ax3a.tick_params(axis='x', rotation=25)

# (b) 表2 各4小时充放电统计
labels_p2 = [item[0] for item in four_hour_intervals]
ch_vals = [np.sum(E_ch[st:ed]) for _, st, ed in four_hour_intervals]
dis_vals = [np.sum(E_dis[st:ed]) for _, st, ed in four_hour_intervals]

x_idx = np.arange(len(labels_p2))
w = 0.35
b1 = ax3b.bar(x_idx - w/2, ch_vals, width=w, label='充电量 (微网充入储能)', color='#9b59b6', edgecolor='#2c3e50', linewidth=1)
b2 = ax3b.bar(x_idx + w/2, dis_vals, width=w, label='放电量 (储能释放微网)', color='#2ecc71', edgecolor='#2c3e50', linewidth=1)

for bar in b1:
    h = bar.get_height()
    if h > 50:
        ax3b.text(bar.get_x() + bar.get_width()/2., h + 100, f'{h:.1f}', ha='center', va='bottom', fontsize=9)
for bar in b2:
    h = bar.get_height()
    if h > 50:
        ax3b.text(bar.get_x() + bar.get_width()/2., h + 100, f'{h:.1f}', ha='center', va='bottom', fontsize=9)

ax3b.set_title('(b) 表2 储能系统各4小时区间充放电量对比', fontsize=12, fontweight='bold', pad=12)
ax3b.set_xticks(x_idx)
ax3b.set_xticklabels(labels_p2, rotation=25)
ax3b.set_ylabel('电量 (kWh)', fontsize=11, fontweight='bold')
ax3b.set_ylim(0, max(max(ch_vals), max(dis_vals)) * 1.18)
ax3b.grid(True, axis='y', linestyle=':', alpha=0.6)
ax3b.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)

plt.tight_layout()
fig3.savefig('figure3_interval_statistics.png', dpi=300)
print("[OK] 图3已保存: figure3_interval_statistics.png")
print("================== 全部流程执行完毕 ==================")