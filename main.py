import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Генерация хаотического временного ряда (логистическое отображение)
def generate_chaotic_series(length=1000, r=3.9, x0=0.5):
    series = np.zeros(length)
    series[0] = x0
    for t in range(1, length):
        series[t] = r * series[t - 1] * (1 - series[t - 1])
    return series


# Создание z-векторов
def create_patterned_z_vectors(series, pattern):
    data, targets = [], []

    for i in range(len(series) - sum(pattern) - 1):
        z_vector = [series[i]]  # Берём значения по паттерну
        dd = 0
        for trm in pattern:
            dd += trm
            z_vector.append(series[i + dd])
        data.append(z_vector)
        targets.append(series[i + dd + 1])  # Следующая точка после последнего элемента паттерна

    return np.array(data), np.array(targets)


# Обновление кластеризации
def update_clusters(X_train, y_train, dbscan_eps=0.2, dbscan_min_samples=5):
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    cluster_labels = dbscan.fit_predict(X_train)
    unique_labels = set(cluster_labels)
    cluster_means = {label: np.mean(np.array(y_train)[np.array(cluster_labels) == label])
                     for label in unique_labels if label != -1}
    return dbscan, cluster_labels, cluster_means


# Определение трудно прогнозируемых точек с помощью RG-DBSCAN
def is_anomalous_point(X_train, eps=0.2, min_samples=5):
    recent_X = np.array(X_train)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(recent_X)
    # Прогнозируемая точка будет аномальной, если DBSCAN классифицирует её как шум (-1)
    return labels[-1] == -1


# Главный алгоритм
def chaotic_forecasting(series, pattern, dbscan_eps=0.2, dbscan_min_samples=5):
    # Формируем обучающую выборку
    z_vectors, y = create_patterned_z_vectors(series, pattern)

    split_idx = int(len(z_vectors) * 0.8)
    X_train, X_test = list(z_vectors[:split_idx]), list(z_vectors[split_idx:])
    y_train, y_test = list(y[:split_idx]), list(y[split_idx:])

    # Первичная кластеризация
    dbscan, cluster_labels, cluster_means = update_clusters(X_train, y_train, dbscan_eps, dbscan_min_samples)

    # Инициализация Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Прогнозирование
    y_pred = []
    count = 0
    for i in range(len(X_test)):
        x = X_test[i]

        if len(cluster_means) > 0:
            closest_cluster = min(cluster_means.keys(), key=lambda k: np.linalg.norm(
                x - np.mean(np.array(X_train)[np.array(cluster_labels) == k], axis=0)))
            predicted_value = cluster_means[closest_cluster]
        else:
            predicted_value = np.mean(y_train)  # Запасной вариант

        y_pred.append(predicted_value)
        y_train.append(predicted_value)
        X_train.append(x)

        # Определяем аномалии с помощью RG-DBSCAN
        if is_anomalous_point(X_train, dbscan_eps, dbscan_min_samples):
            count += 1
            y_pred[-1] = rf_model.predict([x])[0]  # Используем Random Forest для сложных точек
            y_train[-1] = y_pred[-1]


        # Обновляем кластеризацию
        dbscan, cluster_labels, cluster_means = update_clusters(X_train, y_train, dbscan_eps, dbscan_min_samples)

        # Дообучаем Random Forest
        rf_model.fit(X_train, y_train)

    print("Количество непрогнозируемых точек: ", count)
    return y_test, np.array(y_pred)

# Генерация данных и прогноз
series = generate_chaotic_series(1000)
pattern = [1, 2, 2, 3]  # Интервалы выборки точек
y_real, y_forecasted = chaotic_forecasting(series, pattern)

# Вычисление ошибки предсказания
mse = mean_squared_error(y_real, y_forecasted)
mae = mean_absolute_error(y_real, y_forecasted)

print(f"Среднеквадратичная ошибка (MSE): {mse:.5f}")
print(f"Среднеабсолютная ошибка (MAE): {mae:.5f}")
