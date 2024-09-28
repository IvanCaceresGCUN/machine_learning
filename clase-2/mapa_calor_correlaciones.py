import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Leer el archivo Excel
df = pd.read_csv('/Users/ivan.caceres/Documents/data/seattle-weather.csv')

# Mostrar las primeras filas para entender la estructura de los datos
print(df.head())

# Seleccionar solo las columnas numéricas para la correlación
df_numeric = df[['precipitation', 'temp_max', 'temp_min', 'wind']]

# Calcular la matriz de correlaciones
correlation_matrix = df_numeric.corr()

# Crear el mapa de calor utilizando Seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Agregar título y mostrar el gráfico
plt.title('Mapa de Calor de Correlaciones')
plt.tight_layout()
plt.show()




