import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from joblib import delayed, Parallel
from tqdm import tqdm
from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.observables import rmse

# 特征列
features = ['com_x', 'com_y', 'com_z', 'trunk_pitch', 'trunk_roll', 'left_x', 'left_y',
            'right_x', 'right_y', 'left_ankle_pitch', 'left_ankle_roll', 'left_hip_pitch',
            'left_hip_roll', 'left_hip_yaw', 'left_knee', 'right_ankle_pitch',
            'right_ankle_roll', 'right_hip_pitch', 'right_hip_roll',
            'right_hip_yaw', 'right_knee']

prediction = ['fallen']

# 文件路径
files = glob.glob("./experiments/*")

# 多进程读取CSV
def load_files(file_list):
    with Parallel(n_jobs=-1) as parallel:
        dfs = parallel(
            delayed(pd.read_csv)(f, compression="gzip", header=0, sep=",") 
            for f in tqdm(file_list)
        )
    return dfs

# 绘图
def plot_robot(Y, Y_train, F):
    plt.figure(figsize=(10, 7))
    plt.plot(Y_train[1], label="Objective")
    plt.plot(Y[1], label="Fall indicator")
    plt.plot(F[1], label="Applied force")
    plt.legend()
    plt.show()

def plot_robot_results(y_test, y_pred):
    for y_t, y_p in zip(y_test, y_pred):
        if y_t.max() > 0.5:
            plt.figure(figsize=(7, 5))
            plt.plot(y_t, label="Objective")
            plt.plot(y_p, label="Prediction")
            plt.legend()
            plt.show()
            break

# 分批生成器
def batch_generator(X, y, batch_size):
    n_samples = X.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]

if __name__ == "__main__":
    # 加载数据
    dfs = load_files(files)

    # 拼接
    X_all = np.concatenate([df[features].values for df in dfs], axis=0)
    Y_all = np.concatenate([np.roll(df[prediction].values, -500) for df in dfs], axis=0)

    # Optional: Zero out tail after shift
    Y_all[-500:] = 0.0

    # 可视化示例
    F = [df["force_magnitude"].values for df in dfs]
    Y_orig = [df[prediction].values for df in dfs]
    Y_shifted = [np.roll(y, -500) for y in Y_orig]
    plot_robot(Y_orig, Y_shifted, F)

    # 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )

    # 创建ESN（单进程）
    reservoir = Reservoir(300, lr=0.5, sr=0.99, input_bias=False)
    readout = Ridge(1, ridge=1e-3)
    esn = ESN(reservoir=reservoir, readout=readout, workers=1)

    # 分批训练
    batch_size = 5000
    print("Start training in batches...")
    for i, (X_batch, y_batch) in enumerate(batch_generator(X_train, y_train, batch_size)):
        print(f"Training batch {i + 1}")
        esn = esn.fit(X_batch, y_batch)

    # 预测
    res = esn.run(X_test)

    # RMSE
    score = rmse(y_test, res)
    print(f"RMSE: {score:.4f}")

    # 阈值化RMSE
    res_thresh = res.copy()
    res_thresh[res_thresh > 0.5] = 1.0
    res_thresh[res_thresh <= 0.5] = 0.0
    score_thresh = rmse(y_test, res_thresh)
    print(f"RMSE (threshold): {score_thresh:.4f}")

    # 绘制结果
    plot_robot_results([y_test], [res])
