
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Генерация хаотического временного ряда (логистическое отображение)
def generate_chaotic_series(length=1000):
    # Параметры системы Лоренца
    sigma = 10
    rho = 28
    beta = 8 / 3
    dt = 0.01
    steps = length

    # Начальные условия
    x, y, z = 1.0, 0.0, 0.0

    # Массивы для хранения данных
    X_lorenz, Y_lorenz, Z_lorenz = [], [], []

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

    return X_lorenz


def generate_subsequences(series, mn, mx):
    subsequences = []
    for el1 in range(mn, mx + 1):
        for el2 in range(mn, mx + 1):
            for el3 in range(mn, mx + 1):
                for el4 in range(mn, mx + 1):
                    subsequences.append([])
                    for i in range(0, len(series)-el1-el2-el3-el4):
                        subsequences[-1].append([series[i], series[i+el1], series[i+el1+el2],
                                             series[i+el1+el2+el3], series[i+el1+el2+el3+el4]])
    return subsequences

def find_motives(current_pattern, subsequences, max_dif=1000000, top_k=3):
    distances = []
    for m in subsequences:
        sm = 0
        for i in range(4):
            sm += (current_pattern[i] - m[i])**2
        sm = sm
        distances.append([sm, m[4]])
    sorted(distances)
    res = []
    for i in range(top_k):
        if distances[i][0] > max_dif:
            break
        res.append(distances[i][1])
    return res

def find_all_forecast_values(series, subsequences, mn, mx):
    ind = 0
    forecast_values = []
    for el1 in range(mn, mx + 1):
        for el2 in range(mn, mx + 1):
            for el3 in range(mn, mx + 1):
                for el4 in range(mn, mx + 1):
                    if (series[len(series)-el4-el3-el2-el1] is None or series[len(series) - el4 - el3 - el2]
                            is None or series[len(series)-el4-el3] is None or series[len(series)-el4] is None):
                        continue
                    current_pattern = [series[len(series)-el4-el3-el2-el1], series[len(series)-el4-el3-el2], series[len(series)-el4-el3], series[len(series)-el4]]
                    res = find_motives(current_pattern, subsequences[ind])
                    ind += 1
                    for val in res:
                        forecast_values.append(val)
    return forecast_values


def predict_next(forecast_values, delt, epsilon = 0.01, min_samples = 5):
    if len(forecast_values) == 0:
        return None
    if (max(forecast_values) - min(forecast_values) > delt):
        return None
    forecast_values = np.array(forecast_values).reshape(-1, 1)
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(forecast_values)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    return np.mean(forecast_values)


print("Enter row length:")
n = int(input())
print("Min step for pattern:")
mn = int(input())
print("Max step for pattern:")
mx = int(input())
print("How many points need to be predicted:")
cnt = int(input())

a = generate_chaotic_series(n - cnt-50)
b = generate_chaotic_series(n)
v = generate_subsequences(a, mn, mx)
sm = 0
for i in range(n - cnt - 50, n - cnt):
    forecast_values = find_all_forecast_values(a, v, mn, mx)
    if len(forecast_values) != 0:
        sm += (max(forecast_values) - min(forecast_values))
    a.append(b[i])
sm /= 50
for _ in range(cnt):
    forecast_values = find_all_forecast_values(a, v, mn, mx)
    el = predict_next(forecast_values, sm)
    a.append(el)

mae_arr = []
mse_arr = []
u_arr = []

for i in range(n-cnt, n):
    print(a[i], b[i])
    maeNow = 0
    mseNow = 0
    unpredictable = 0
    for j in range(n - cnt, i + 1):
        if a[i] is None:
            unpredictable += 1
            continue
        maeNow += abs(a[i] - b[i])
        mseNow += (a[i] - b[i]) ** 2
    u_arr.append(unpredictable)
    maeNow /= (i + 1 - (n - cnt) + 1)
    mseNow /= (i + 1 - (n - cnt) + 1)
    mae_arr.append(maeNow)
    mse_arr.append(mseNow)

plt.figure(figsize=(10, 6))  # Размер графика

# Рисуем все три линии
plt.plot(mae_arr, label="MAE", color='blue')
plt.plot(mse_arr, label="MSE", color='green')
plt.plot(u_arr, label="Unpredictable Points", color='red')

# Подписи и оформление
plt.title("Сравнение метрик: MAE, MSE и непрогнозируемые точки")
plt.xlabel("Итерация")
plt.ylabel("Значение")
plt.legend()  # Легенда
plt.grid(True)
plt.tight_layout()

plt.show()
