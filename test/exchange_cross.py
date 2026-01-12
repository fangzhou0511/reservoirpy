import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import os

# -----------------------
# 1. 加载数据
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
# 2. 合并
# -----------------------
data = aud_cny.rename(columns={"收盘":"AUD_CNY"})\
    .merge(aud_usd.rename(columns={"收盘":"AUD_USD"}), on="日期")\
    .merge(gbp_cny.rename(columns={"收盘":"GBP_CNY"}), on="日期")\
    .merge(gbp_usd.rename(columns={"收盘":"GBP_USD"}), on="日期")\
    .merge(usd_cny.rename(columns={"收盘":"USD_CNY"}), on="日期")

# -----------------------
# 3. 构造交叉特征
# -----------------------
cross = pd.DataFrame()
cross["日期"] = data["日期"]
cross["AUD_CNY"] = data["AUD_CNY"]

# 比值
cross["R_AUDUSD_USDCNY"] = data["AUD_USD"] / data["USD_CNY"]
cross["R_GBPCNY_USDCNY"] = data["GBP_CNY"] / data["USD_CNY"]
cross["R_AUDUSD_GBPUSD"] = data["AUD_USD"] / data["GBP_USD"]

# 差值
cross["D_AUDCNY_USDCNY"] = data["AUD_CNY"] - data["USD_CNY"]
cross["D_GBPCNY_USDCNY"] = data["GBP_CNY"] - data["USD_CNY"]

# -----------------------
# 4. 删除极端或无效行
cross = cross.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# -----------------------
# 5. 数据分割
train_idx = (cross["日期"] >= "2022-01-01") & (cross["日期"] < "2025-01-01")
test_idx = (cross["日期"] >= "2025-01-01")

train_df = cross.loc[train_idx].reset_index(drop=True)
test_df = cross.loc[test_idx].reset_index(drop=True)

# -----------------------
# 6. 归一化
feat_cols = [c for c in cross.columns if c != "日期"]

min_vals = train_df[feat_cols].min()
max_vals = train_df[feat_cols].max()

train_scaled = 2 * (train_df[feat_cols] - min_vals) / (max_vals - min_vals) - 1
test_scaled = 2 * (test_df[feat_cols] - min_vals) / (max_vals - min_vals) - 1

# -----------------------
# 7. 窗口特征
window_size = 5

def create_windows(data, target_col=0):
    X, y = [], []
    arr = data.values
    for i in range(len(data)-window_size):
        window = arr[i:i+window_size,:].flatten()
        X.append(window)
        y.append(arr[i+window_size,target_col])
    return np.array(X), np.array(y).reshape(-1,1)

X_train, y_train = create_windows(train_scaled, target_col=0)
X_test, y_test = create_windows(test_scaled, target_col=0)

# -----------------------
# 8. 循环训练10次
result_file = "cny_aud_esn_results_cross.csv"

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
    
    r2 = rsquare(y_test, y_pred)
    nrmse_val = nrmse(y_test, y_pred)
    
    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")
    
    with open(result_file,"a") as f:
        f.write(f"{i},{r2:.6f},{nrmse_val:.6f}\n")
    
    plt.figure(figsize=(12,5))
    plt.plot(y_test, "--", label="True normalized")
    plt.plot(y_pred, "-", label="ESN prediction")
    plt.plot(np.abs(y_test - y_pred), label="Absolute deviation")
    plt.xlabel("Days in 2025")
    plt.ylabel("Normalized Close Price")
    plt.title(f"CNY/AUD Cross Feature Prediction (Iter {i})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cny_aud_cross_run_{i}.png")
    plt.close()

print("\n========== All runs finished ==========")
print(f"Results saved to {result_file}")
