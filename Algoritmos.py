import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.svm import SVC  # Para clasificación
from sklearn.model_selection import GridSearchCV
from scipy.stats import t

# Leer el archivo Excel
df = pd.read_csv(r'D:\Universidad\Cun\Diplomado Machine Learning en Python\seattle-weather.csv')

# Convertir 'date' a formato datetime
df['date'] = pd.to_datetime(df['date'])

# Extraer componentes de la fecha
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Eliminar la columna 'date' después de extraer los componentes
df = df.drop(columns=['date'])

# Verificar los tipos de datos
print(df.dtypes)

# Imprimir los nombres de las columnas
print("Nombres de las columnas en el DataFrame:")
print(df.columns)

# Mostrar las primeras filas del DataFrame
#print(df.head())
#print (df.describe())
#print (df.dtypes)
#print (df.isnull().sum())

''' 
plt.title('Distribución de la Columna Numérica')
plt.xlabel('Valores')
plt.xlabel('Frecuencia')
plt.show()
'''

# Funciones

def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""    
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

plt.style.use('ggplot')

# Aquí no necesitas cargar un nuevo DataFrame con Seaborn
print(df.sample(5))  # Muestra 5 muestras aleatorias del DataFrame original
#sns.boxplot(x='precipitation', y='temp_max', data=df)
#plt.show() """

""" sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data=df, x="precipitation", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=df, x="temp_max", kde=True, ax=axs[0, 1], color='red')
sns.histplot(data=df, x="temp_min", kde=True, ax=axs[1, 0], color='skyblue')
sns.histplot(data=df, x="wind", kde=True, ax=axs[1, 1], color='orange')

plt.show() """

# Filtrar solo los días soleados
sunny_days = df[df['weather'] == 'sunny']

# Supongamos que tienes una columna llamada 'temperature'
# Calcular la media y desviación estándar de la temperatura en días soleados
sunny_mean = sunny_days['temp_max'].mean()
sunny_std = sunny_days['temp_max'].std()

# Tamaño de la muestra
n = len(sunny_days)

# Nivel de confianza (95% por defecto)
confidence_level = 0.95

# Grados de libertad
df_sunny = n - 1

# Valor crítico de t
t_critical = t.ppf((1 + confidence_level) / 2, df_sunny)

# Margen de error
margin_of_error = t_critical * (sunny_std / np.sqrt(n))

# Límite inferior y superior del intervalo de confianza
lower_bound = sunny_mean - margin_of_error
upper_bound = sunny_mean + margin_of_error

# Imprimir el intervalo de confianza
print(f"Intervalo de confianza al {confidence_level*100}%:")
print(f"Límite inferior: {lower_bound:.2f}")
print(f"Límite superior: {upper_bound:.2f}")

# Crear un boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='weather', y='temp_max', data=df)

# Añadir líneas para el intervalo de confianza
plt.axhline(y=lower_bound, color='r', linestyle='--', label='Límite Inferior IC')
plt.axhline(y=upper_bound, color='g', linestyle='--', label='Límite Superior IC')

# Añadir título y leyenda
plt.title('Boxplot de Temperatura por Estado del Clima')
plt.xlabel('Estado del Clima')
plt.ylabel('temp_max')
plt.legend()

'''# Mostrar la gráfica
plt.show()'''

# Supongamos que tienes un DataFrame llamado 'df' con tus datos
X = df.drop(columns=['weather'])  # Elimina la columna que quieres predecir (e.g., 'target')
y = df['weather']  # La columna objetivo (e.g., temperatura o clase climática)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de clasificación
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Predicción
y_pred_clf = classifier.predict(X_test_scaled)

# Evaluación
accuracy = accuracy_score(y_test, y_pred_clf)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred_clf))

# Entrenar un modelo de clasificación SVC
svc = SVC(kernel='rbf')  # También puedes probar otros kernels: 'linear', 'poly', etc.
svc.fit(X_train_scaled, y_train)

# Predicción
y_pred_svc = svc.predict(X_test_scaled)

# Evaluación del modelo SVC
accuracy = accuracy_score(y_test, y_pred_svc)
print(f'Accuracy (SVC): {accuracy}')
print(classification_report(y_test, y_pred_svc))

# Definir los hiperparámetros para probar
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'linear']
}

# Para clasificación (SVC)
grid_svc = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_svc.fit(X_train_scaled, y_train)

# Mejor modelo encontrado
print(grid_svc.best_params_)

