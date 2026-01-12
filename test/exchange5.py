import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import os

# 滚动预测 + Ensemble
# -----------------------
# 1. 参数配置
# -----------------------
UNITS = 200
RIDGE = 1e-2
SR = 0.9
LR = 0.3
WINDOW_SIZE = 2
N_ENSEMBLE = 2

# -----------------------
# 2. 数据读取
# -----------------------
df = pd.read_csv("exchange_rate/AUD_CNY历史数据.csv")
df.columns = df.columns.str.strip()
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

# -----------------------
# 3. 数据分割
# -----------------------
train_df = df[df["日期"] < "2025-01-01"]
test_df = df[df["日期"] >= "2025-01-01"]

train_close = train_df["收盘"].values
test_close = test_df["收盘"].values

# -----------------------
# 4. 归一化
min_val = train_close.min()
max_val = train_close.max()

def scale(x):
    return 2 * (x - min_val) / (max_val - min_val) - 1

def inverse_scale(x):
    return (x + 1) / 2 * (max_val - min_val) + min_val

train_scaled = scale(train_close)
test_scaled = scale(test_close)

# -----------------------
# 5. 创建窗口
def create_windows(data):
    X, y = [], []
    for i in range(len(data) - WINDOW_SIZE):
        X.append(data[i:i+WINDOW_SIZE])
        y.append(data[i+WINDOW_SIZE])
    return np.array(X), np.array(y).reshape(-1,1)

# -----------------------
# 6. 滚动 Ensemble 预测
print("\n========== Rolling Ensemble Forecast Start ==========")
rolling_preds = []
rolling_truth = []

# 初始化历史数据
history = list(train_scaled)

for i in range(len(test_scaled)):
    # 每一步 ensemble
    preds_this_step = []

    for j in range(N_ENSEMBLE):
        hist_array = np.array(history)
        X_train, y_train = create_windows(hist_array)

        seed_value = np.random.randint(0,10000)
        reservoir = Reservoir(
            units=UNITS,
            sr=SR,
            lr=LR,
            input_scaling=1.0,
            rc_connectivity=0.1,
            input_connectivity=0.2,
            seed=seed_value
        )
        readout = Ridge(ridge=RIDGE)

        esn = reservoir >> readout
        esn.fit(X_train, y_train)

        # 预测
        X_input = np.array(history[-WINDOW_SIZE:]).reshape(1, -1)
        y_pred = esn.run(X_input)
        preds_this_step.append(y_pred.item())

    # 平均10个预测
    mean_pred = np.mean(preds_this_step)
    rolling_preds.append(mean_pred)
    rolling_truth.append(test_scaled[i])

    # 更新历史
    history.append(test_scaled[i])

    print(f"Day {i+1}/{len(test_scaled)} done.")

# -----------------------
# 7. 评估
y_pred_inv = inverse_scale(np.array(rolling_preds))
y_truth_inv = inverse_scale(np.array(rolling_truth))

r2 = rsquare(np.array(rolling_truth).reshape(-1,1), np.array(rolling_preds).reshape(-1,1))
nrmse_val = nrmse(np.array(rolling_truth).reshape(-1,1), np.array(rolling_preds).reshape(-1,1))

print(f"\n========== Rolling Ensemble Forecast Finished ==========")
print(f"R²: {r2:.4f}")
print(f"NRMSE: {nrmse_val:.4f}")

# -----------------------
# 8. 输出结果文件
result_file = "cny_aud_rolling_ensemble_results.csv"
if os.path.exists(result_file):
    os.remove(result_file)

result_df = pd.DataFrame({
    "Day": np.arange(1,len(test_scaled)+1),
    "True": y_truth_inv,
    "Predicted": y_pred_inv,
    "AbsError": np.abs(y_truth_inv - y_pred_inv)
})
result_df.to_csv(result_file, index=False)
print(f"Saved results to {result_file}")

# -----------------------
# 9. 绘图
plt.figure(figsize=(12,5))
plt.plot(y_truth_inv, "--", label="True value")
plt.plot(y_pred_inv, "-", label="Rolling Ensemble Prediction")
plt.plot(np.abs(y_truth_inv - y_pred_inv), label="Absolute Deviation")
plt.xlabel("Days in 2025")
plt.ylabel("CNY/AUD Close Price")
plt.title("Rolling Ensemble Forecast of CNY/AUD Exchange Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("cny_aud_rolling_ensemble.png")
plt.close()

print("Saved rolling ensemble forecast plot.")
