import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('C:/Users/Admin/Desktop/Programacion/datascience1/dataset/tv_shows.csv')

# Inspección inicial
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nDescripción estadística de las columnas numéricas:")
print(df.describe())

print("Columnas del DataFrame:", df.columns)

# Contar y visualizar valores nulos
missing_values = df.isnull().sum()
print("\nValores nulos por columna:")
print(missing_values)

plt.figure(figsize=(10, 5))
missing_values.plot(kind='bar', color='salmon')
plt.title('Valores Nulos por Columna')
plt.xlabel('Columnas')
plt.ylabel('Cantidad de Valores Nulos')
plt.show()

# Limpiar la columna 'IMDb' (extraer solo la parte numérica antes del '/')
df['IMDb'] = df['IMDb'].str.extract('(\d+\.\d+)').astype(float)  # Extrae el número antes del '/'

# Limpiar la columna 'Rotten Tomatoes' (extraer solo la parte numérica antes del '/')
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].str.extract('(\d+)').astype(float)  # Extrae el número antes del '/'

# Tratamiento de nulos en la columna "Age"
if 'Age' in df.columns:
    # Utilizamos la moda para rellenar valores nulos en "Age" si es categórica
    Age_mode = df['Age'].mode()[0]  # La moda (valor más frecuente) de la columna "Age"
    df['Age'] = df['Age'].fillna(Age_mode)
    
    # Verificar que no hay más valores nulos en "Age"
    print("\nValores nulos en 'Age' después del tratamiento:", df['Age'].isnull().sum())

# Visualización del resultado después del tratamiento de nulos en "Age" y "IMDb"
plt.figure(figsize=(10, 5))
sns.heatmap(df[['Age', 'IMDb']].isnull(), cbar=False, cmap='viridis')
plt.title('Mapa de Calor de Valores Nulos (Después de Rellenar)')
plt.show()

# Contar la cantidad de títulos por plataforma (sólo si estas columnas existen y son binarias)
platform_columns = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
if all(column in df.columns for column in platform_columns):
    platform_counts = df[platform_columns].sum()
    
    # Gráfico de barras
    plt.figure(figsize=(8, 5))
    platform_counts.plot(kind='bar', color='skyblue')
    plt.title('Cantidad de Títulos por Plataforma')
    plt.xlabel('Plataforma')
    plt.ylabel('Número de Títulos')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("\nUna o más columnas de plataforma no están presentes o no son binarias.")

# Limpiar la columna 'Year' (convertir a numérico)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Limpiar la columna 'Age' (extraer el número antes del '+' y convertir a numérico)
df['Age'] = df['Age'].str.extract('(\d+)').astype(float)  # Extrae solo los números

# Verificar los primeros valores del DataFrame después de la limpieza
print(df.head())

# Crear una nueva columna que indique la plataforma
df_melted = df.melt(id_vars=['Year', 'Age'], value_vars=['Netflix', 'Hulu', 'Prime Video', 'Disney+'],
                    var_name='Plataforma', value_name='Pertenece')

# Filtrar solo las filas donde 'Pertenece' es 1 (es decir, los títulos pertenecen a esa plataforma)
df_melted = df_melted[df_melted['Pertenece'] == 1]

# Crear gráfico de dispersión
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_melted, x='Year', y='Age', hue='Plataforma', palette='Set1', s=100, alpha=0.7)

# Personalización
plt.title('Relación entre Año de Lanzamiento y Edad Recomendada por Plataforma')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Edad Recomendada')
plt.legend(title='Plataforma', bbox_to_anchor=(1.05, 1), loc='upper left')

# Mostrar gráfico
plt.tight_layout()
plt.show()

# Crear una nueva columna que indique la plataforma
df_melted = df.melt(id_vars=['IMDb', 'Rotten Tomatoes'], value_vars=['Netflix', 'Hulu', 'Prime Video', 'Disney+'],
                    var_name='Plataforma', value_name='Pertenece')

# Filtrar solo las filas donde 'Pertenece' es 1 (es decir, los títulos pertenecen a esa plataforma)
df_melted = df_melted[df_melted['Pertenece'] == 1]

# Crear la figura con dos subgráficos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot para IMDb
sns.boxplot(x='Plataforma', y='IMDb', data=df_melted, ax=axes[0], palette='Set1')
axes[0].set_title('Distribución de IMDb por Plataforma')
axes[0].set_xlabel('Plataforma')
axes[0].set_ylabel('IMDb')

# Boxplot para Rotten Tomatoes
sns.boxplot(x='Plataforma', y='Rotten Tomatoes', data=df_melted, ax=axes[1], palette='Set2')
axes[1].set_title('Distribución de Rotten Tomatoes por Plataforma')
axes[1].set_xlabel('Plataforma')
axes[1].set_ylabel('Rotten Tomatoes')

# Ajustar el diseño para que los gráficos no se superpongan
plt.tight_layout()
plt.show()

print("fin")

