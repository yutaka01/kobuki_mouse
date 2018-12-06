import numpy as np
from numpy import linalg

u = np.array([3, 4])
v = np.array([-4, 3])

i = np.inner(u, v)
n = linalg.norm(u) * linalg.norm(v)

c = i / n

print(c)
a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

print(a)

