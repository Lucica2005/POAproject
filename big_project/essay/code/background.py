import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置数据
# 互联网用户增长数据
internet_data = {
    'Year': [2010, 2012, 2014, 2016, 2018, 2020],
    'Internet Users (millions)': [2000, 2580, 2800, 3600, 3800, 4500]
}

# 短视频平台用户增长数据
video_data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Douyin': [10, 30, 80, 150, 300, 500],
    'Bilibili': [20, 40, 70, 130, 200, 300],
    'Kuaishou': [5, 20, 50, 100, 200, 400]
}

# 网络平台收益增长数据
revenue_data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Douyin Revenue': [100, 200, 400, 600, 900, 1200],
    'Bilibili Revenue': [80, 160, 320, 480, 700, 950],
    'Kuaishou Revenue': [70, 140, 280, 420, 650, 850]
}

df_internet = pd.DataFrame(internet_data)
df_video = pd.DataFrame(video_data)
df_revenue = pd.DataFrame(revenue_data)

# 创建图形和子图
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# 设置子图之间的间距
fig.subplots_adjust(hspace=0.5)

# 互联网用户增长图
sns.lineplot(x='Year', y='Internet Users (millions)', data=df_internet, ax=axs[0], color='#929ea6',linewidth=0.8,marker='o', 
             markersize=10, markerfacecolor='orange', markeredgewidth=2)
axs[0].set_title('Growth of Internet Users Over Time')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Internet Users (millions)')
axs[0].grid(True)

# 短视频平台用户增长图
sns.lineplot(x='Year', y='Douyin', data=df_video, ax=axs[1], label='Douyin', color='red',linewidth=0.5)
sns.lineplot(x='Year', y='Bilibili', data=df_video, ax=axs[1], label='Bilibili', color='green',linewidth=0.5)
sns.lineplot(x='Year', y='Kuaishou', data=df_video, ax=axs[1], label='Kuaishou', color='purple',linewidth=0.5)
axs[1].set_title('User Growth of Short Video Platforms')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Users (millions)')
axs[1].legend()
axs[1].grid(True)

# 网络平台收益增长图
sns.lineplot(x='Year', y='Douyin Revenue', data=df_revenue, ax=axs[2], label='Douyin Revenue', color='red',linewidth=0.5)
sns.lineplot(x='Year', y='Bilibili Revenue', data=df_revenue, ax=axs[2], label='Bilibili Revenue', color='green',linewidth=0.5)
sns.lineplot(x='Year', y='Kuaishou Revenue', data=df_revenue, ax=axs[2], label='Kuaishou Revenue', color='purple',linewidth=0.5)
axs[2].set_title('Revenue Growth of Network Platforms')
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Revenue (millions USD)')
axs[2].legend()
axs[2].grid(True)

# 显示图形
plt.show()
