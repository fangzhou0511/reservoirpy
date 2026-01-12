import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import os

# -----------------------
# 1. 读取所有数据
# -----------------------
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df[["日期", "收盘"]]

aud_cny = load_data("exchange_rate/AUD_CNY历史数据.csv")
aud_usd = load_data("exchange_rate/AUD_USD历史数据.csv")
gbp_cny = load_data("exchange_rate/GBP_CNY历史数据.csv")
gbp_usd = load_data("exchange_rate/GBP_USD历史数据.csv")
usd_cny = load_data("exchange_rate/USD_CNY历史数据.csv")

# -----------------------
# 2. 按日期合并
# -----------------------
data = aud_cny.rename(columns={"收盘":"AUD_CNY"})\
    .merge(aud_usd.rename(columns={"收盘":"AUD_USD"}), on="日期")\
    .merge(gbp_cny.rename(columns={"收盘":"GBP_CNY"}), on="日期")\
    .merge(gbp_usd.rename(columns={"收盘":"GBP_USD"}), on="日期")\
    .merge(usd_cny.rename(columns={"收盘":"USD_CNY"}), on="日期")

# -----------------------
# 3. 数据分割
# -----------------------
train_df = data[(data["日期"] >= "2022-01-01") & (data["日期"] < "2025-01-01")]
test_df = data[data["日期"] >= "2025-01-01"]

# -----------------------
# 4. 归一化
# -----------------------
min_vals = train_df.iloc[:,1:].min()
max_vals = train_df.iloc[:,1:].max()

train_scaled = 2 * (train_df.iloc[:,1:] - min_vals) / (max_vals - min_vals) - 1
test_scaled = 2 * (test_df.iloc[:,1:] - min_vals) / (max_vals - min_vals) - 1

# -----------------------
# 5. 窗口特征
# -----------------------
window_size = 5

def create_multi_windows(data, target_col):
    X, y = [], []
    arr = data.values
    for i in range(len(data)-window_size):
        feat = []
        for j in range(arr.shape[1]):  # 每一列
            feat.extend(arr[i:i+window_size,j])
        X.append(feat)
        y.append(arr[i+window_size,target_col])  # target列
    return np.array(X), np.array(y).reshape(-1,1)

# 预测AUD_CNY（index=0）
X_train, y_train = create_multi_windows(train_scaled, target_col=0)
X_test, y_test = create_multi_windows(test_scaled, target_col=0)

# -----------------------
# 6. 循环训练10次
# -----------------------
result_file = "cny_aud_esn_results_multivariate.csv"

if os.path.exists(result_file):
    os.remove(result_file)

with open(result_file, "w") as f:
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
    
    # 反归一化
    y_pred_inv = (y_pred + 1) /2 * (max_vals["AUD_CNY"] - min_vals["AUD_CNY"]) + min_vals["AUD_CNY"]
    y_test_inv = (y_test + 1) /2 * (max_vals["AUD_CNY"] - min_vals["AUD_CNY"]) + min_vals["AUD_CNY"]
    
    r2 = rsquare(y_test, y_pred)
    nrmse_val = nrmse(y_test, y_pred)
    
    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")
    
    with open(result_file,"a") as f:
        f.write(f"{i},{r2:.6f},{nrmse_val:.6f}\n")
    
    plt.figure(figsize=(12,5))
    plt.plot(y_test_inv, "--", label="True value")
    plt.plot(y_pred_inv, "-", label="ESN prediction")
    plt.plot(np.abs(y_test_inv - y_pred_inv), label="Absolute deviation")
    plt.xlabel("Days in 2025")
    plt.ylabel("CNY/AUD Close Price")
    plt.title(f"CNY/AUD Prediction (Multivariate Input, Iteration {i})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cny_aud_multivariate_run_{i}.png")
    plt.close()

print("\n========== All runs finished ==========")
print(f"Results saved to {result_file}")
