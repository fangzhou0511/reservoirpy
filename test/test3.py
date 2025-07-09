from reservoirpy.datasets import mackey_glass
from reservoirpy.datasets import to_forecasting
from reservoirpy.observables import nrmse, rsquare
from test2 import plot_train_test, plot_results
import matplotlib.pyplot as plt
import numpy as np


timesteps = 2510
tau = np.random.randint(10, 25)
X = mackey_glass(timesteps, tau=tau, seed=np.random.randint(0, 100000))
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

x, y = to_forecasting(X, forecast=100)
X_train2, y_train2 = x[:2000], y[:2000]
X_test2, y_test2 = x[2000:], y[2000:]

plot_train_test(X_train2, y_train2, X_test2, y_test2)

units = 300
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-8
seed = None

def reset_esn():
    from reservoirpy.nodes import Reservoir, Ridge

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity)
    readout   = Ridge(1, ridge=regularization)

    return reservoir >> readout

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout


y_pred2 = esn.fit(X_train2, y_train2).run(X_test2)


plot_results(y_pred2, y_test2, sample=400)

rsquare(y_test2, y_pred2), nrmse(y_test2, y_pred2)