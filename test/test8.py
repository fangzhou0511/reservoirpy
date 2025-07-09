from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from reservoirpy.datasets import japanese_vowels
from reservoirpy import set_seed, verbosity
from reservoirpy.observables import nrmse, rsquare

from sklearn.metrics import accuracy_score

set_seed(42)
verbosity(0)


X_train, Y_train, X_test, Y_test = japanese_vowels()

plt.figure()
plt.imshow(X_train[0].T, vmin=-1.2, vmax=2)
plt.title(f"A sample vowel of speaker {np.argmax(Y_train[0]) +1}")
plt.xlabel("Timesteps")
plt.ylabel("LPC (cepstra)")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(X_train[50].T, vmin=-1.2, vmax=2)
plt.title(f"A sample vowel of speaker {np.argmax(Y_train[50]) +1}")
plt.xlabel("Timesteps")
plt.ylabel("LPC (cepstra)")
plt.colorbar()
plt.show()

sample_per_speaker = 30
n_speaker = 9
X_train_per_speaker = []

for i in range(n_speaker):
    X_speaker = X_train[i*sample_per_speaker: (i+1)*sample_per_speaker]
    X_train_per_speaker.append(np.concatenate(X_speaker).flatten())

plt.boxplot(X_train_per_speaker)
plt.xlabel("Speaker")
plt.ylabel("LPC (cepstra)")
plt.show()

# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.
X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

from reservoirpy.nodes import Reservoir, Ridge, Input

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

model = [source >> reservoir, source] >> readout


Y_pred = model.fit(X_train, Y_train, stateful=False, warmup=2).run(X_test, stateful=False)

Y_pred_class = [np.argmax(y_p, axis=1) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t, axis=1) for y_t in Y_test]

score = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(Y_pred_class, axis=0))

print("Accuracy: ", f"{score * 100:.3f} %")

X_train, Y_train, X_test, Y_test = japanese_vowels()

from reservoirpy.nodes import Reservoir, Ridge, Input


source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

model = source >> reservoir >> readout

states_train = []
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states[-1, np.newaxis])

readout.fit(states_train, Y_train)

Y_pred = []
for x in X_test:
    states = reservoir.run(x, reset=True)
    y = readout.run(states[-1, np.newaxis])
    Y_pred.append(y)

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t) for y_t in Y_test]

score = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")