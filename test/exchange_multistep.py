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
# 4. 多步目标生成
window_size = 5
future_steps = 5

def create_multi_step(data):
    X, y = [], []
    for i in range(len(data)-window_size-future_steps):
        X.append(data[i:i+window_size])
        future = data[i+window_size:i+window_size+future_steps]
        y.append(future.mean())
    return np.array(X), np.array(y).reshape(-1,1)

X_train, y_train = create_multi_step(train_scaled)
X_test, y_test = create_multi_step(test_scaled)

# -----------------------
# 5. 循环训练
result_file = "cny_aud_esn_results_multistep.csv"

if os.path.exists(result_file):
    os.remove(result_file)

with open(result_file,"w") as f:
    f.write("Run,R2,NRMSE\n")

for i in range(1,11):
    print(f"\n========== Training iteration {i} ==========")
    
    reservoir = Reservoir(
        units=300,
        sr=0.9,
        lr=0.3,
        input_scaling=1.0,
        rc_connectivity=0.1,
        input_connectivity=0.2,
        seed=np.random.randint(0,10000)
    )
    readout = Ridge(ridge=1e-2)
    
    esn = reservoir >> readout
    esn.fit(X_train, y_train)
    
    y_pred = esn.run(X_test)
    
    r2 = rsquare(y_test, y_pred)
    nrmse_val = nrmse(y_test, y_pred)
    
    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")
    
    with open(result_file,"a") as f:
        f.write(f"{i},{r2:.6f},{nrmse_val:.6f}\n")
    
    plt.figure(figsize=(12,5))
    plt.plot(y_test, "--", label="True normalized mean")
    plt.plot(y_pred, "-", label="ESN prediction")
    plt.plot(np.abs(y_test - y_pred), label="Absolute deviation")
    plt.xlabel("Steps")
    plt.ylabel("Normalized mean price")
    plt.title(f"CNY/AUD Multi-step Mean Prediction (Iter {i})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cny_aud_multistep_run_{i}.png")
    plt.close()

print("\n========== All runs finished ==========")
print(f"Results saved to {result_file}")
