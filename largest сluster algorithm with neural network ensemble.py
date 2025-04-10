
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor

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

def find_motives(current_pattern, subsequences, max_dif=80, top_k=3):
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


def predict_next(forecast_values, epsilon = 0.448, min_samples = 5):
    if len(forecast_values) == 0:
        return None
    forecast_values = np.array(forecast_values).reshape(-1, 1)
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(forecast_values)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        return None  # Все точки — шум

        # Сортировка кластеров по убыванию размера
    sorted_indices = np.argsort(counts)[::-1]
    counts_sorted = counts[sorted_indices]
    labels_sorted = unique_labels[sorted_indices]

    # Проверка условия: самый большой кластер как минимум в 3 раза больше второго
    if len(counts_sorted) > 1 and counts_sorted[0] >= 3 * counts_sorted[1]:
        largest_label = labels_sorted[0]
        largest_cluster_values = forecast_values[labels == largest_label]
        return np.mean(largest_cluster_values)
    elif len(counts_sorted) == 1:
        # Если только один кластер, возвращаем его среднее
        largest_label = labels_sorted[0]
        largest_cluster_values = forecast_values[labels == largest_label]
        return np.mean(largest_cluster_values)
    else:
        return None


print("Enter row length:")
n = int(input())
print("Min step for pattern:")
mn = int(input())
print("Max step for pattern:")
mx = int(input())
print("How many points need to be predicted:")
cnt = int(input())

a = generate_chaotic_series(n - cnt)
b = generate_chaotic_series(n)
v = generate_subsequences(a, mn, mx)

# Обучение ансамбля лин рег
n_points = n-cnt
x_series = np.array(generate_chaotic_series(n_points))

window_size = 5
X = []
y = []

for i in range(n_points - window_size):
    X.append(x_series[i:i + window_size])
    y.append(x_series[i + window_size])

X = np.array(X)
y = np.array(y)

base_mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
ensemble_mlp = BaggingRegressor(base_mlp, n_estimators=10, random_state=42)
ensemble_mlp.fit(X, y)
last_window = x_series[-window_size:].tolist()

for _ in range(cnt):
    forecast_values = find_all_forecast_values(a, v, mn, mx)
    el = predict_next(forecast_values)
    if el is None:
        el = ensemble_mlp.predict([last_window])[0]
    a.append(el)
    last_window = last_window[1:] + [el]

mae_arr = []
mse_arr = []
u_arr = []

for i in range(n-cnt, n):
    print(a[i], b[i])
    maeNow = 0
    mseNow = 0
    unpredictable = 0
    for j in range(n - cnt, i + 1):
        if a[j] is None:
            unpredictable += 1
            continue
        maeNow += abs(a[j] - b[j])
        mseNow += (a[j] - b[j]) ** 2
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
