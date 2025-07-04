from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse, rsquare
import matplotlib.pyplot as plt

### Step 1: Load the dataset

X = mackey_glass(n_timesteps=2000)  # (2000, 1)-shaped array
# create y by shifting X, and train/test split
x_train, x_test, y_train, y_test = to_forecasting(X, test_size=0.5)

### Step 2: Create an Echo State Network

# 100 neurons reservoir, spectral radius = 1.25, leak rate = 0.3
reservoir = Reservoir(units=100, sr=1.25, lr=0.3)
# feed-forward layer of neurons, trained with L2-regularization
readout = Ridge(ridge=1e-5)
# connect the two nodes
esn = reservoir >> readout

### Step 3: Fit, run and evaluate the ESN

esn.fit(x_train, y_train, warmup=100)
predictions = esn.run(x_test)

print(f"RMSE: {rmse(y_test, predictions)}; R^2 score: {rsquare(y_test, predictions)}")
# RMSE: 0.0020282; R^2 score: 0.99992
plt.plot(y_test, label="True")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from reservoirpy.datasets import mackey_glass

# # 生成时间序列
# X = mackey_glass(n_timesteps=2000)

# # 提取一维数组
# x = X.flatten()

# # 创建画布
# plt.figure(figsize=(15,10))

# # 子图1: 整个时间序列
# plt.subplot(2,2,1)
# plt.plot(x)
# plt.title("Full Mackey-Glass Time Series")
# plt.xlabel("Time step")
# plt.ylabel("Value")

# # 子图2: 前200步放大细节
# plt.subplot(2,2,2)
# plt.plot(x[:200], color='green')
# plt.title("First 200 Steps (Zoom In)")
# plt.xlabel("Time step")
# plt.ylabel("Value")

# # 子图3: 序列的直方图（值分布）
# plt.subplot(2,2,3)
# plt.hist(x, bins=50, color='orange', edgecolor='k')
# plt.title("Histogram of Values")
# plt.xlabel("Value")
# plt.ylabel("Frequency")

# # 子图4: 延迟嵌入相空间轨迹
# # (x(t), x(t - tau))
# tau = 20
# plt.subplot(2,2,4)
# plt.plot(x[:-tau], x[tau:], lw=0.5)
# plt.title(f"Phase Space: x(t) vs x(t - {tau})")
# plt.xlabel("x(t)")
# plt.ylabel(f"x(t - {tau})")

# plt.tight_layout()
# plt.show()
