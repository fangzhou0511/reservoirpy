from reservoirpy.datasets import to_forecasting
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import nrmse, rsquare
from test2 import plot_train_test, plot_results
import matplotlib.pyplot as plt
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge

timesteps = 2510
# tau = 17
tau = np.random.randint(10, 25)
X = mackey_glass(timesteps, tau=tau, seed=np.random.randint(0, 100000))
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
seed = 1234


from reservoirpy.nodes import FORCE

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)


esn_online = reservoir >> readout

x, y = to_forecasting(X, forecast=10)
X_train1, y_train1 = x[:2000], y[:2000]
X_test1, y_test1 = x[2000:], y[2000:]

outputs_pre = np.zeros(X_train1.shape)
for t, (x, y) in enumerate(zip(X_train1, y_train1)): # for each timestep of training data:
    outputs_pre[t, :] = esn_online.train(x, y)

plot_results(outputs_pre, y_train1, sample=100)

plot_results(outputs_pre, y_train1, sample=500)

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)


esn_online = reservoir >> readout

esn_online.train(X_train1, y_train1)

pred_online = esn_online.run(X_test1)  # Wout est maintenant figée


plot_results(pred_online, y_test1, sample=500)

rsquare(y_test1, pred_online), nrmse(y_test1, pred_online)