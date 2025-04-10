import numpy as np
import matplotlib.pyplot as plt

# Параметры системы Лоренца
sigma = 10
rho = 28
beta = 8 / 3
dt = 0.01
steps = 10000

x, y, z = 1.0, 0.0, 0.0

X_lorenz, Y_lorenz, Z_lorenz = [], [], []

# Метод Эйлера для интегрирования системы Лоренца
for _ in range(steps):
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt

    x += dx
    y += dy
    z += dz

    X_lorenz.append(x)
    Y_lorenz.append(y)
    Z_lorenz.append(z)

# Построение графиков для системы Лоренца (только изменения по x)
plt.figure(figsize=(10, 6))
plt.plot(X_lorenz[:1000], label='x', alpha=0.7)
plt.title("Хаотический ряд для системы Лоренца")
plt.xlabel("Шаг")
plt.ylabel("Значения переменных")
plt.legend()
plt.grid(True)
plt.show()

# Параметры для логистического отображения
r = 3.99
x_logistic = 0.5
iterations = 50

X_logistic = []

# Генерация значений с использованием логистического отображения
for _ in range(iterations):
    x_logistic = r * x_logistic * (1 - x_logistic)
    X_logistic.append(x_logistic)

plt.figure(figsize=(10, 6))
plt.plot(X_logistic, label="x_n (логистическое отображение)", color='r', alpha=0.7)
plt.title("Хаотический ряд для логистического отображения")
plt.xlabel("Итерация")
plt.ylabel("x_n")
plt.legend()
plt.grid(True)
plt.show()
