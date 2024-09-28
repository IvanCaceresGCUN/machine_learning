import pandas as pd
import numpy as np
import scipy.stats as stats

# Cargar el archivo CSV con los datos del clima
data = pd.read_csv("seattle-weather.csv")

# Limpiar los datos: eliminar filas con valores faltantes en las columnas numéricas
data_clean = data[['precipitation', 'temp_max', 'temp_min', 'wind']].dropna()

# 1. Conceptos estadísticos fundamentales: estadísticas descriptivas
media_temp_max = np.mean(data_clean['temp_max'])
mediana_temp_max = np.median(data_clean['temp_max'])
desv_std_temp_max = np.std(data_clean['temp_max'])

print(f"Media de la temperatura máxima: {media_temp_max}")
print(f"Mediana de la temperatura máxima: {mediana_temp_max}")
print(f"Desviación estándar de la temperatura máxima: {desv_std_temp_max}")

# 2. Análisis inferencial: intervalo de confianza para la media
confidence_level = 0.95
mean_temp = np.mean(data_clean['temp_max'])
sem_temp = stats.sem(data_clean['temp_max'])  # Error estándar de la media
ci_temp = stats.t.interval(confidence_level, len(data_clean['temp_max'])-1, loc=mean_temp, scale=sem_temp)

print(f"Intervalo de confianza al 95% para la temperatura máxima: {ci_temp}")

# 3. Pruebas estadísticas: prueba t de Student
# Comparar la media de la temperatura máxima con un valor de referencia, por ejemplo 15 grados
t_stat, p_value = stats.ttest_1samp(data_clean['temp_max'], 15)

print(f"Estadístico t: {t_stat}")
print(f"Valor p: {p_value}")

if p_value < 0.05:
    print("Rechazamos la hipótesis nula: la media de la temperatura máxima es significativamente diferente de 15°C")
else:
    print("No podemos rechazar la hipótesis nula: la media de la temperatura máxima no es significativamente diferente de 15°C")

# Guardar los resultados en un archivo
with open("resultados_analisis.txt", "w") as f:
    f.write(f"Media de la temperatura máxima: {media_temp_max}\n")
    f.write(f"Mediana de la temperatura máxima: {mediana_temp_max}\n")
    f.write(f"Desviación estándar de la temperatura máxima: {desv_std_temp_max}\n")
    f.write(f"Intervalo de confianza al 95%: {ci_temp}\n")
    f.write(f"Estadístico t: {t_stat}\n")
    f.write(f"Valor p: {p_value}\n")
