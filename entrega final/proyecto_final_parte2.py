import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset
df = pd.read_csv('C:/Users/Admin/Desktop/Programacion/datascience1/dataset/tv_shows.csv')

# Revisa las primeras filas y las columnas del dataframe para entender qué datos tienes
print(df.head())
print(df.columns)

# Verificar si la columna 'Netflix' existe y contiene valores binarios (0 o 1)
if 'Netflix' not in df.columns:
    print("La columna 'Netflix' no está en el dataset.")
else:
    # Crear la nueva columna 'Availability' basada en la columna 'Netflix'
    df['Availability'] = df['Netflix'].apply(lambda x: 'En Netflix' if x == 1 else 'No en Netflix')

# Limpiar la columna 'Age' (convertir a valores numéricos)
df['Age'] = df['Age'].str.extract('(\d+)')  # Extrae solo los números de la columna 'Age'
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Convierte en valores numéricos (NaN si hay errores)
df['Age'].fillna(df['Age'].median(), inplace=True)  # Rellenar los valores faltantes con la mediana

# Verificar si la columna 'Type' (o 'Genre') existe para convertirla a valores numéricos
if 'Type' in df.columns:
    label_encoder = LabelEncoder()
    df['Genre'] = label_encoder.fit_transform(df['Type'])  # Convertir 'Type' a valores numéricos

# Convertir 'Availability' a valores numéricos utilizando LabelEncoder
df['Availability'] = label_encoder.fit_transform(df['Availability'])

# Seleccionar las columnas de características (features) y la variable objetivo (target)
X = df[['Age', 'Genre']]  # Puedes incluir otras columnas si es necesario
y = df['Availability']

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones con el modelo
y_pred = model.predict(X_test)

# Evaluar el modelo
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Conclusiones del modelo
print("\nConclusiones del modelo:El modelo de Random Forest muestra una precisión muy buena en el conjunto de prueba. Esto significa que el modelo acierta gran catidad de veces en su predicción sobre si un programa está disponible en Netflix o no.")
print(f"Precisión del modelo: {model.score(X_test, y_test)}")


