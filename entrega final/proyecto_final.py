# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset
df = pd.read_csv('C:/Users/Admin/Desktop/Programacion/datascience1/dataset/tv_shows.csv')


# --- i) Limpieza y Exploración de Datos ---
# Previsualizar los datos
print(df.info())
print(df.head())

# Normalizar y gestionar valores nulos en la columna 'Age'
df['Age'] = df['Age'].fillna('Unknown')

# Codificar variables categóricas
age_mapping = {age: i for i, age in enumerate(df['Age'].unique())}
df['Age'] = df['Age'].map(age_mapping)

# --- ii) Selección de Características ---
# Variables independientes y objetivo (si está disponible en Netflix)
X = df[['Year', 'Age', 'Hulu', 'Prime Video', 'Disney+']]
y = df['Netflix']

# Seleccionar las mejores K características
k_best = SelectKBest(score_func=f_classif, k=3)
X_new = k_best.fit_transform(X, y)
selected_features = k_best.get_support(indices=True)
print("Características seleccionadas:", X.columns[selected_features])

# --- iii) Entrenamiento de un Modelo de Clasificación ---
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Modelo: Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# --- iv) Cálculo de Métricas ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# --- v) Generar Conclusiones ---
print("Conclusiones:")
print("- El modelo obtuvo una precisión del {:.2f}, lo que indica un buen desempeño general.".format(accuracy))
print("- Las características seleccionadas parecen ser relevantes para predecir la disponibilidad en Netflix.")
print("- Se podría mejorar el modelo ajustando hiperparámetros o probando otros algoritmos.")
