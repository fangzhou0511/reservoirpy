import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import os

# -----------------------
# 1. 数据读取
# -----------------------
df = pd.read_csv("exchange_rate/AUD_CNY历史数据.csv")
df.columns = df.columns.str.strip()
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
min_val = train_close.min()
max_val = train_close.max()

train_scaled = 2 * (train_close - min_val) / (max_val - min_val) - 1
test_scaled = 2 * (test_close - min_val) / (max_val - min_val) - 1

# -----------------------
# 4. 窗口定义
window_size = 2
n_seeds = 10

# -----------------------
# 5. 结果文件
result_file = "cny_aud_esn_bestconfig.csv"
if os.path.exists(result_file):
    os.remove(result_file)

with open(result_file, "w") as f:
    f.write("Run,R2,NRMSE\n")

# -----------------------
# 6. 创建窗口
def create_windows(data):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y).reshape(-1,1)

X_train, y_train = create_windows(train_scaled)
X_test, y_test = create_windows(test_scaled)

predictions_all = []

# -----------------------
# 7. 循环训练
for i in range(1, n_seeds+1):
    print(f"\n========== Seed Run {i} ==========")

    seed_value = np.random.randint(0,10000)

    reservoir = Reservoir(
        units=200,
        sr=0.9,
        lr=0.3,
        input_scaling=1.0,
        rc_connectivity=0.1,
        input_connectivity=0.2,
        seed=seed_value
    )
    readout = Ridge(ridge=1e-2)

    esn = reservoir >> readout
    esn.fit(X_train, y_train)

    y_pred = esn.run(X_test)
    predictions_all.append(y_pred)

    # 反归一化
    y_pred_inv = (y_pred + 1) / 2 * (max_val - min_val) + min_val
    y_test_inv = (y_test + 1) / 2 * (max_val - min_val) + min_val

    r2 = rsquare(y_test, y_pred)
    nrmse_val = nrmse(y_test, y_pred)

    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")

    with open(result_file, "a") as f:
        f.write(f"{i},{r2:.6f},{nrmse_val:.6f}\n")

    plt.figure(figsize=(12,5))
    plt.plot(y_test_inv, "--", label="True value")
    plt.plot(y_pred_inv, "-", label="ESN prediction")
    plt.plot(np.abs(y_test_inv - y_pred_inv), label="Absolute deviation")
    plt.xlabel("Days in 2025")
    plt.ylabel("CNY/AUD Close Price")
    plt.title(f"CNY/AUD Prediction Window=2 Seed={i}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cny_aud_bestconfig_seed{i}.png")
    plt.close()

# -----------------------
# 8. Ensemble平均
y_pred_mean = np.mean(predictions_all, axis=0)
y_pred_inv = (y_pred_mean + 1) / 2 * (max_val - min_val) + min_val
y_test_inv = (y_test + 1) / 2 * (max_val - min_val) + min_val

r2_mean = rsquare(y_test, y_pred_mean)
nrmse_mean = nrmse(y_test, y_pred_mean)

print(f"\n=== Ensemble Mean ===")
print(f"R²: {r2_mean:.4f}")
print(f"NRMSE: {nrmse_mean:.4f}")

with open(result_file, "a") as f:
    f.write(f"ensemble,{r2_mean:.6f},{nrmse_mean:.6f}\n")

plt.figure(figsize=(12,5))
plt.plot(y_test_inv, "--", label="True value")
plt.plot(y_pred_inv, "-", label="Ensemble prediction")
plt.plot(np.abs(y_test_inv - y_pred_inv), label="Absolute deviation")
plt.xlabel("Days in 2025")
plt.ylabel("CNY/AUD Close Price")
plt.title(f"CNY/AUD Ensemble Prediction (Best Config)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"cny_aud_bestconfig_ensemble.png")
plt.close()

print("\n========== All runs finished ==========")
print(f"Results saved to {result_file}")
