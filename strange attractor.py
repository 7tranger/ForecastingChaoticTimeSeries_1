import numpy as np
import matplotlib.pyplot as plt

# Параметры системы Лоренца
sigma = 10
rho = 28
beta = 8 / 3

x, y, z = 1.0, 0.0, 0.0
dt = 0.01
steps = 10000

X, Y, Z = [], [], []

# Итерации метода Эйлера
for _ in range(steps):
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt

    x += dx
    y += dy
    z += dz

    X.append(x)
    Y.append(y)
    Z.append(z)

# Построение графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, lw=0.5)
ax.set_title("Странный аттрактор Лоренца")
plt.show()
