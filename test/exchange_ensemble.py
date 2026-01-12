import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import os

# -----------------------
# 1. 加载数据
# -----------------------
df = pd.read_csv("exchange_rate/AUD_CNY历史数据.csv")
df.columns = df.columns.str.strip()
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

# -----------------------
# 2. 分割
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
# 4. 单币种窗口特征
window_size = 5
future_steps = 5

def create_single_step(data):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y).reshape(-1,1)

def create_multi_step(data):
    X, y = [], []
    for i in range(len(data)-window_size-future_steps):
        X.append(data[i:i+window_size])
        future = data[i+window_size:i+window_size+future_steps]
        y.append(future.mean())
    return np.array(X), np.array(y).reshape(-1,1)

# 为了对齐长度：multi-step先
X_train_m, y_train_m = create_multi_step(train_scaled)
X_test_m, y_test_m = create_multi_step(test_scaled)

# single-step相同索引区间
start_idx = window_size
end_idx = window_size + len(y_test_m)

X_train_s = []
y_train_s = []
X_test_s = []
y_test_s = []

s_X_train_all, s_y_train_all = create_single_step(train_scaled)
s_X_test_all, s_y_test_all = create_single_step(test_scaled)

X_train_s = s_X_train_all[:len(y_train_m)]
y_train_s = s_y_train_all[:len(y_train_m)]
X_test_s = s_X_test_all[:len(y_test_m)]
y_test_s = s_y_test_all[:len(y_test_m)]

# -----------------------
# 5. 循环训练融合
result_file = "cny_aud_esn_results_ensemble.csv"

if os.path.exists(result_file):
    os.remove(result_file)

with open(result_file,"w") as f:
    f.write("Run,R2,NRMSE\n")

for i in range(1,11):
    print(f"\n========== Training iteration {i} ==========")

    # Single-step ESN
    reservoir_s = Reservoir(
        units=300,
        sr=0.9,
        lr=0.3,
        input_scaling=1.0,
        rc_connectivity=0.1,
        input_connectivity=0.2,
        seed=np.random.randint(0,10000)
    )
    readout_s = Ridge(ridge=1e-2)
    esn_s = reservoir_s >> readout_s
    esn_s.fit(X_train_s, y_train_s)
    pred_s = esn_s.run(X_test_s)

    # Multi-step ESN
    reservoir_m = Reservoir(
        units=300,
        sr=0.9,
        lr=0.3,
        input_scaling=1.0,
        rc_connectivity=0.1,
        input_connectivity=0.2,
        seed=np.random.randint(0,10000)
    )
    readout_m = Ridge(ridge=1e-2)
    esn_m = reservoir_m >> readout_m
    esn_m.fit(X_train_m, y_train_m)
    pred_m = esn_m.run(X_test_m)

    # 融合
    pred_ensemble = (pred_s + pred_m) / 2

    r2 = rsquare(y_test_s, pred_ensemble)
    nrmse_val = nrmse(y_test_s, pred_ensemble)

    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")

    with open(result_file,"a") as f:
        f.write(f"{i},{r2:.6f},{nrmse_val:.6f}\n")

    plt.figure(figsize=(12,5))
    plt.plot(y_test_s, "--", label="True normalized")
    plt.plot(pred_ensemble, "-", label="Ensemble prediction")
    plt.plot(np.abs(y_test_s - pred_ensemble), label="Absolute deviation")
    plt.xlabel("Steps")
    plt.ylabel("Normalized Close Price")
    plt.title(f"CNY/AUD Ensemble Prediction (Iter {i})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cny_aud_ensemble_run_{i}.png")
    plt.close()

print("\n========== All runs finished ==========")
print(f"Results saved to {result_file}")
