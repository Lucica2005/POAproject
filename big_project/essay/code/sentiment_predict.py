import pandas as pd
import matplotlib.pyplot as plt

# 给定的比例数据
ratios = [1.0, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 1.1, 1.2, 2.3]

# 计算每个时间点的正负面百分比
positive_ratios = [r / (r + 1) * 100 for r in ratios]
negative_ratios = [100 - pr for pr in positive_ratios]

# 创建时间序列数据框
time_points = list(range(1, len(ratios) + 1))
data = pd.DataFrame({
    'Time': time_points,
    'Positive': positive_ratios,
    'Negative': negative_ratios
})

# 绘制线图
colors = ['#B8A9C9', '#F5CAC3']  # 莫兰迪色系的紫色、肉色

fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制正负面百分比的线图
ax1.plot(data['Time'], data['Positive'], label='Positive (%)', color=colors[0], linewidth=0.8, marker='o', markersize=5, markerfacecolor='lightgray', markeredgewidth=2)
ax1.plot(data['Time'], data['Negative'], label='Negative (%)', color=colors[1], linewidth=0.8, marker='o', markersize=5, markerfacecolor='lightgray', markeredgewidth=2)

ax1.set_xlabel('Time')
ax1.set_ylabel('Percentage')
ax1.set_ylim(0, 100)
ax1.legend(loc='upper left')

plt.title('Sentiment Analysis Over Time')
plt.show()
