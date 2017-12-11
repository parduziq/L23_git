import numpy as np
import matplotlib.pyplot as plt
a = np.load("/Users/qendresa/Desktop/L23/Simulation/Syn_dynamics_11_12_2017/outRC.npy")


print(min(a), max(a), np.mean(a), a.shape)
plt.hist(a, bins = "auto")
plt.show()
