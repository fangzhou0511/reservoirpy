import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare

# -----------------------
# 1. 数据读取
# -----------------------
df = pd.read_csv("exchange_rate/AUD_CNY历史数据.csv")
df.columns = df.columns.str.strip()  # 去除列名空格

# 日期解析
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

# -----------------------
# 2. 数据分割
# -----------------------
train_df = df[(df["日期"] >= "2022-01-01") & (df["日期"] < "2025-01-01")]
test_df = df[df["日期"] >= "2025-01-01"]

train_close = train_df["收盘"].values
test_close = test_df["收盘"].values

# -----------------------
# 3. 归一化
# -----------------------
min_val = train_close.min()
max_val = train_close.max()

train_scaled = 2 * (train_close - min_val) / (max_val - min_val) - 1
test_scaled = 2 * (test_close - min_val) / (max_val - min_val) - 1

# -----------------------
# 4. 窗口输入生成
# -----------------------
window_size = 5

def create_windows(data):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y).reshape(-1,1)

X_train, y_train = create_windows(train_scaled)
X_test, y_test = create_windows(test_scaled)

# -----------------------
# 5. 创建ESN
# -----------------------
reservoir = Reservoir(
    units=200,
    sr=0.9,
    lr=0.3,
    input_scaling=1.0,
    rc_connectivity=0.1,
    input_connectivity=0.2,
    seed=42
)
readout = Ridge(ridge=1e-2)

esn = reservoir >> readout

# -----------------------
# 6. 训练
# -----------------------
esn.fit(X_train, y_train)

# -----------------------
# 7. 预测
# -----------------------
y_pred = esn.run(X_test)

# -----------------------
# 8. 反归一化
# -----------------------
y_pred_inv = (y_pred + 1) / 2 * (max_val - min_val) + min_val
y_test_inv = (y_test + 1) / 2 * (max_val - min_val) + min_val

# -----------------------
# 9. 绘图
# -----------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_inv, "--", label="True value")
plt.plot(y_pred_inv, "-", label="ESN prediction")
plt.plot(np.abs(y_test_inv - y_pred_inv), label="Absolute deviation")
plt.xlabel("Days in 2025")
plt.ylabel("CNY/AUD Close Price")
plt.title("CNY/AUD Exchange Rate Prediction (ESN with Window Input)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("cny_aud_prediction_window.png")
plt.show()

# -----------------------
# 10. 评估
# -----------------------
print("R²:", rsquare(y_test, y_pred))
print("NRMSE:", nrmse(y_test, y_pred))
