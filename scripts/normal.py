import numpy as np
import matplotlib.pyplot as plt
# 生成正态分布的数据
size = 10000  # 数据点数量
data_min = 0
data_max = 1000
# 绘制直方图
# 绘制正态分布曲线
x = np.linspace(data_min, data_max, 1000)
mean = 800  # 均值
std = 10  # 标准差
y1 = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
mean = 210  # 均值
std = 50  # 标准差
y2 = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
alpha = 0.2
y3 = y2 * alpha + y1 * (1 - alpha)
plt.plot(x, y1, color='red', linewidth=2, label='In Clear')
# plt.plot(x, y2, color='blue', linewidth=2, label='Caused by Weather Element')
# plt.plot(x, y3, color='green', linewidth=2, label='Result')
# 添加标题和标签
plt.title('Distribution of distances in one of angles ')
plt.xlabel('distance')
plt.ylabel('Probability Density')
plt.legend()  # 添加图例
plt.grid(True)  # 添加网格线
plt.savefig('./normal.png')  # 保存图像
plt.show()  # 显示图像