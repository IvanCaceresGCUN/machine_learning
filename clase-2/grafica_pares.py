import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Leer el archivo Excel
df = pd.read_csv('/Users/ivan.caceres/Documents/data/seattle-weather.csv')


# Mostrar las primeras filas para entender la estructura de los datos
print(df.head())

# Seleccionar las columnas numéricas y categóricas relevantes
df_numeric = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']]

# Crear la gráfica de pares (pair plot)
plt.figure(figsize=(10, 6))
sns.pairplot(df_numeric, hue='weather', palette='Set2', diag_kind='kde')

# Mostrar el gráfico
plt.suptitle('Gráfica de Pares de Variables Climáticas', y=1.02)  # Título con ajuste
plt.tight_layout()
plt.show()
