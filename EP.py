import splines as sp
import numpy as np
import matplotlib.pyplot as plt

# a função original
n_weights = 10
w = np.random.randn(n_weights)
x = np.arange(0, 15.01, 0.01)
s = sp.spline(w, x.min(), x.max())
y = s(x)
label1,  = plt.plot(x, y, c='g')

# criação de amostras com ruídos
x = np.arange(0, 15.5, 0.5)
n_samples = x.shape[0]
noise = np.random.normal(loc=0.0, scale=20, size=n_samples)
y = s(x) + noise
label2 = plt.scatter(x, y, c='r', marker='x')

# matriz B
B = np.zeros((n_samples, n_weights))
for i in range(n_samples):
    for j in range(n_weights):
        B[i, j] = s.beta_j(j, x[i])

# suavizador
M2 = sp.matrix_m2(n_weights)

b = B.T.dot(y)
M1 = B.T.dot(B)

lamb = 1
w = np.linalg.solve(M1 + lamb * M2, b)
s = sp.spline(w, x.min(), x.max())
label3, = plt.plot(x, s(x), c='b')

plt.legend((label1, label2, label3), ('original', 'samples', 'predicted'))
plt.show()
