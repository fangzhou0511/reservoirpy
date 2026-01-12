import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from sklearn.decomposition import PCA
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
# 3. 计算差分
# -----------------------
diff_data = data.iloc[:,1:].diff().dropna()
diff_dates = data["日期"].iloc[1:]  # 对应日期

# -----------------------
# 4. 数据分割
# -----------------------
train_idx = (diff_dates >= "2022-01-01") & (diff_dates < "2025-01-01")
test_idx = (diff_dates >= "2025-01-01")

train_df = diff_data.loc[train_idx]
test_df = diff_data.loc[test_idx]

# -----------------------
# 5. 标准化
# -----------------------
mean_vals = train_df.mean()
std_vals = train_df.std()

train_scaled = (train_df - mean_vals) / std_vals
test_scaled = (test_df - mean_vals) / std_vals

# -----------------------
# 6. PCA降维
# -----------------------
pca = PCA(n_components=5)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)

# -----------------------
# 7. 窗口特征
# -----------------------
window_size = 5

def create_windows(data, target_col=0):
    X, y = [], []
    for i in range(len(data)-window_size):
        window = data[i:i+window_size,:].flatten()
        X.append(window)
        y.append(data[i+window_size, target_col])
    return np.array(X), np.array(y).reshape(-1,1)

X_train, y_train = create_windows(train_pca)
X_test, y_test = create_windows(test_pca)

# -----------------------
# 8. 循环训练10次
# -----------------------
result_file = "cny_aud_esn_results_diff_pca.csv"

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
    plt.plot(y_test, "--", label="True diff")
    plt.plot(y_pred, "-", label="ESN prediction")
    plt.plot(np.abs(y_test - y_pred), label="Absolute deviation")
    plt.xlabel("Days in 2025")
    plt.ylabel("Differenced Close Price")
    plt.title(f"CNY/AUD Diff Prediction (PCA+Window, Iteration {i})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cny_aud_diff_pca_run_{i}.png")
    plt.close()

print("\n========== All runs finished ==========")
print(f"Results saved to {result_file}")
