import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


np.random.seed(4)
x = np.random.normal(0.0, 0.5, 100)
xy = np.c_[x, -0.03 * np.ones_like(x)]
print(xy.shape)
plt.plot(xy[:, 0], xy[:, 1], "*", color='orange')
plt.hist(x, bins=15, color='purple', normed=True, ec='black')
t = np.linspace(np.min(x), np.max(x), 100)
plt.plot(t, norm.pdf(t, loc=0.0, scale=0.5), color='red')
plt.axvline(x=0.0, linestyle='--', color='cyan')
plt.savefig('MLE_Discrete.jpg')
plt.show()
