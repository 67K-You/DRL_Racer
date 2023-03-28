import os
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), "logs.txt"), "r") as f:
    lines = [map(float, line.split()) for line in f.readlines()]
loss_q, loss_pi = zip(*lines)

plt.figure()
plt.plot(np.arange(len(loss_q)), np.log(loss_q))
plt.xlabel("Iterations")
plt.ylabel("Loss Q value")
plt.show()

plt.figure()
plt.plot(np.arange(len(loss_pi)), np.log(loss_pi))
plt.xlabel("Iterations")
plt.ylabel("Loss Pi action")
plt.show()