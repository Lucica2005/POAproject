import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# Data to plot
labels = ['空间', '外观', '满意', '大', '动力', '油耗', '车', '好', '有点', '和', '内饰', '感觉', '还是', '可以', '配置', '座椅', '后排', '发动机', '这个', '非常']
sizes = [0.356689435309765, 0.27276421699117376, 0.2507015515044462, 0.22561978442479805, 0.20129279274338002, 0.19409339663718472, 0.19400630716815817, 0.18491997256638748, 0.17203073115045717, 0.1645700666371822, 0.15667395477877444, 0.13911091185841895, 0.12967621938054205, 0.11873197610620484, 0.11847070769912517, 0.09803371230089333, 0.09585647557522943, 0.09243095646018488, 0.09025371973452098, 0.08717655849558267]
colors = ['#89b8d6', '#929ea6','#d9b0b0','#5b7dac','#d9ac6b','#bf9180','#9ab099','#8fa699','#d9b0c3','#d9d9d9','#93a6bf','#f2d399','#7185ad','#97a6a0','#b8778d','#7e78a7','#9e9bd2','#c3aae1','#7b5b6d','#8b7d86']  # Simplified Morandi color palette

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=240)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#plt.title('Sentiment Distribution Pie Chart')
plt.show()
