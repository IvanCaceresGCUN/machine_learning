import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
# Leer el archivo Excel
df = pd.read_csv('/Users/ivan.caceres/Documents/data/seattle-weather.csv')

# Mostrar las primeras filas para entender la estructura de los datos
print(df.head())

# Contar el número de días por condición climática
weather_count = df['weather'].value_counts()

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(weather_count.index, weather_count.values, color='blue')
plt.title('Distribución de Condiciones Climáticas')
plt.xlabel('Condición Climática')
plt.ylabel('Número de Días')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
