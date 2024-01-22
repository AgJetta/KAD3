import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Funkcja do skalowania liniowego danych
def linear_scale(data):
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    scaled_data = (data - min_values) / (max_values - min_values)
    return scaled_data

# Funkcja obliczająca odległość euklidesową
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Klasa KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Oblicz odległości
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        return self._predict_with_k(x, distances, self.k)

    def _predict_with_k(self, x, distances, k):
        # Pobierz k najbliższych próbek, ich etykiety i odległości
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        # Ważone głosowanie
        label_weight = {}
        for label, dist in zip(k_nearest_labels, k_nearest_distances):
            if dist != 0:  # Zabezpieczenie przed dzieleniem przez zero
                if label in label_weight:
                    label_weight[label] += 1 / dist
                else:
                    label_weight[label] = 1 / dist

        # Sprawdzenie czy istnieje remis
        if label_weight:
            max_weight = max(label_weight.values())
            best_labels = [label for label, weight in label_weight.items() if weight == max_weight]

            if len(best_labels) > 1:
                # W przypadku remisu, wybierz klasę, która ma łącznie mniejszą odległość
                best_label = min(best_labels, key=lambda label: sum(
                    dist for l, dist in zip(k_nearest_labels, k_nearest_distances) if l == label))
                return best_label

            return best_labels[0]
        else:
            # Jeśli label_weight jest pusty, zwróć dowolną wartość, na przykład pierwszą etykietę
            return k_nearest_labels[0]

        if len(best_labels) > 1:
            # W przypadku remisu, wybierz klasę, która ma łącznie mniejszą odległość
            best_label = min(best_labels, key=lambda label: sum(
                dist for l, dist in zip(k_nearest_labels, k_nearest_distances) if l == label))
            return best_label

        return best_labels[0]

# Załadowanie datasetu
path_test = 'data_test.csv'
path_train = 'data_train.csv'
column_names = ["Długość działki kielicha (cm)", "Szerokość działki kielicha (cm)",
                "Długość płatka (cm)", "Szerokość płatka (cm)", "Gatunek"]



data_test = pd.read_csv(path_test, names=column_names)
data_train = pd.read_csv(path_train, names=column_names)

# Normalizacja danych
X_train = linear_scale(data_train.iloc[:, :-1].values)
y_train = data_train.iloc[:, -1].values
X_test = linear_scale(data_test.iloc[:, :-1].values)
y_test = data_test.iloc[:, -1].values

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

## WSZYSTKIE

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)
# plt.title('Dokładność dla różnych wartości k w k-NN', fontsize=16)
plt.ylim(97, 100.1)  # Ustawienie dolnej granicy na 92%
plt.yticks(np.arange(97, 100.5, 0.5), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)


plt.show()

## 0 1 - długość działki kielicha i szerokość działki kielicha
# Indeksy cech w danej kombinacji
feature_indices = [0, 1]

# Pobierz dane treningowe i testowe z wybranymi cechami
X_train_subset = X_train[:, feature_indices]
X_test_subset = X_test[:, feature_indices]

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test_subset)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)

plt.ylim(65, 75)  # Ustawienie dolnej granicy na 92%
plt.yticks(np.arange(65,76 , 1), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)

plt.show()


## 0 2 - długość działki kielicha i długość płatka
# Indeksy cech w danej kombinacji
feature_indices = [0, 2]

# Pobierz dane treningowe i testowe z wybranymi cechami
X_train_subset = X_train[:, feature_indices]
X_test_subset = X_test[:, feature_indices]

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test_subset)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)

plt.ylim(90, 101)
plt.yticks(np.arange(90,101 , 1), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)

plt.show()

## 0 3 - długość działki kielicha i szerokość płatka
# Indeksy cech w danej kombinacji
feature_indices = [0, 3]

# Pobierz dane treningowe i testowe z wybranymi cechami
X_train_subset = X_train[:, feature_indices]
X_test_subset = X_test[:, feature_indices]

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test_subset)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)

plt.ylim(90, 101)
plt.yticks(np.arange(90,101 , 1), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)

plt.show()


## 1 2 - szerokość działki kielicha i długość płatka
# Indeksy cech w danej kombinacji
feature_indices = [1, 2]

# Pobierz dane treningowe i testowe z wybranymi cechami
X_train_subset = X_train[:, feature_indices]
X_test_subset = X_test[:, feature_indices]

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test_subset)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)

plt.ylim(87, 98)
plt.yticks(np.arange(87,98 , 1), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)

plt.show()

## 1 3 - szerokość działki kielicha i szerokosć płatka
# Indeksy cech w danej kombinacji
feature_indices = [1, 3]

# Pobierz dane treningowe i testowe z wybranymi cechami
X_train_subset = X_train[:, feature_indices]
X_test_subset = X_test[:, feature_indices]

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test_subset)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)

plt.ylim(90, 101)
plt.yticks(np.arange(90,101 , 1), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)

plt.show()


## 2 3 - dlugosc i szerokosć płatka
# Indeksy cech w danej kombinacji
feature_indices = [2, 3]

# Pobierz dane treningowe i testowe z wybranymi cechami
X_train_subset = X_train[:, feature_indices]
X_test_subset = X_test[:, feature_indices]

# Pętla dla różnych wartości k
accuracy_scores = []
k_values = list(range(2, 16))

for k in k_values:
    # Utworzenie i trenowanie modelu k-NN
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train)

    # Testowanie modelu na danych testowych
    predictions = knn.predict(X_test_subset)

    # Obliczenie dokładności
    accuracy = np.mean(predictions == y_test) * 100  # Dokładność jako procent
    accuracy_scores.append(accuracy)

# Rysowanie histogramu dokładności
plt.bar(k_values, accuracy_scores, color='dodgerblue')
plt.xlabel('Liczba sąsiadów, wartość (k)', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)

plt.ylim(90, 101)
plt.yticks(np.arange(90,101 , 1), fontsize=12)
plt.xticks(range(2, 16), fontsize=12)

plt.show()
