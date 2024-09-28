# Análisis de Datos Climáticos

Este proyecto contiene análisis y visualizaciones de datos climáticos utilizando Python, pandas, Matplotlib y Seaborn. Los datos se extraen de un archivo CSV y se utilizan para generar gráficos que facilitan la comprensión de las condiciones climáticas.

## Contenido

- Estadistica descriptiva 
- Gráfico de barras: Distribución de condiciones climáticas
- Mapa de calor de correlaciones
- Gráfica de pares
- Medidas de estadisticas


## Requisitos

- Python 3.x
- pandas
- Matplotlib
- Seaborn
- Numpy
- Scipy.stats
- Scikit-learn

Puedes instalar las dependencias necesarias utilizando pip


## Analisis de base de datos 

Hay 6 variables en este conjunto de datos:

4 variables continuas.
1 variable para la fecha.
1 variable que se refiere al clima.


![Screenshot 2024-09-28 at 10 06 44 AM](https://github.com/user-attachments/assets/8713f48b-13b5-40da-8e88-e31df3370dd3)

## Correlaciones y Análisis de Valores Nulos
Se utiliza un mapa de calor para mostrar la correlación entre las variables numéricas, destacando la relación positiva entre la temperatura máxima y mínima.

![Screenshot 2024-09-28 at 10 09 49 AM](https://github.com/user-attachments/assets/888290af-8208-4ec0-ae58-d2d482ad93af)


## Gráfico de Barras
 

![barras](https://github.com/user-attachments/assets/f8f1f1e2-3236-4b4f-9ae0-9763d93d78df)

## Mapa de Calor de Correlaciones
  

![calor_corelaciones](https://github.com/user-attachments/assets/dd44fa3d-3ca2-4f3a-8fa2-04b1d011609d)

## Gráfica de Pares


![pares](https://github.com/user-attachments/assets/c02486ed-5cda-40d6-b5ad-c5d5210866dd)

## ASIMETRÍA UTILIZANDO DIAGRAMAS DE CAJA

La asimetría (o skewness en inglés) es una medida que describe la simetría de la distribución de datos. Indica si los datos se distribuyen de manera uniforme alrededor de la media. Puede ser:
Asimetría Positiva: Cuando la cola derecha de la distribución es más larga o más gruesa que la izquierda. En este caso, la media es mayor que la mediana.
Asimetría Negativa: Cuando la cola izquierda es más larga o más gruesa que la derecha. Aquí, la media es menor que la mediana.
Diagramas de Caja: Un diagrama de caja (o boxplot) es una representación gráfica que resume un conjunto de datos a través de sus cuartiles. Muestra cinco estadísticas clave:

![diagramas_cajas](https://github.com/user-attachments/assets/6b52e85a-a112-45e8-b4df-17c6e923a752)


Valor mínimo: El valor más bajo.
Primer cuartil (Q1): El valor que separa el 25% inferior de los datos.
Mediana (Q2): El valor que divide el conjunto de datos en dos mitades iguales.
Tercer cuartil (Q3): El valor que separa el 75% superior de los datos.
Valor máximo: El valor más alto.

plt.figure(figsize=(12,6))
sns.boxplot("precipitation","weather",data=data,palette="YlOrBr")
## Conceptos estadísticos fundamentales en ciencia de datos
Se incluyen medidas descriptivas como la media, la mediana y la desviación estándar.
Análisis inferencial y descriptivo con SciPy:
Se utiliza scipy.stats para realizar pruebas inferenciales, como el intervalo de confianza para la media.
Pruebas estadísticas y su aplicación en Python:
Se ejecutan pruebas estadísticas como la t de Student para comparar la media de los datos con un valor de referencia.

Descripción de las secciones:
Estadísticas descriptivas:

Calcula la media, mediana y desviación estándar de la columna temp_max.
Intervalo de confianza:

Usa la función scipy.stats.t.interval para calcular el intervalo de confianza al 95% para la temperatura máxima.
Prueba t de Student:

Compara la media de temp_max con un valor hipotético de 15°C. Si el valor p es menor a 0.05, se rechaza la hipótesis nula de que la media es igual a 15°C.

Uso
Cargar y limpiar los datos: El script carga los datos desde seattle-weather.csv y elimina filas con valores faltantes en las columnas de interés (precipitación, temperatura máxima y mínima, velocidad del viento).

## Estadísticas descriptivas: El script calcula las siguientes estadísticas para la temperatura máxima:

Media
Mediana
Desviación estándar
Análisis inferencial:

Se calcula el intervalo de confianza al 95% para la media de la temperatura máxima usando la distribución t de Student.
Pruebas estadísticas:

Se ejecuta una prueba t de Student para comparar la media de la temperatura máxima con un valor de referencia (15°C en este caso). El resultado de la prueba t indicará si la diferencia es estadísticamente significativa.

Resultados
El script generará un archivo de texto con los siguientes resultados:

![estadistica](https://github.com/user-attachments/assets/f5b1f894-ded4-49ad-8648-a3dc93512089)


Media, mediana y desviación estándar de la temperatura máxima.
Intervalo de confianza al 95% para la media de la temperatura máxima.
Resultado de la prueba t de Student, incluyendo el estadístico t y el valor p.

![resultados](https://github.com/user-attachments/assets/bb377b99-083e-4bbf-9dcb-f6aa87610f3b)


Media de la temperatura máxima: 16.44°C
Mediana de la temperatura máxima: 15.6°C
Desviación estándar de la temperatura máxima: 5.29°C
Intervalo de confianza al 95%: (16.06°C, 16.82°C)
Estadístico t: 5.49
Valor p: 2.1e-07
Rechazamos la hipótesis nula: la media de la temperatura máxima es significativamente diferente de 15°C

## Evaluación del modelo
La búsqueda de hiperparámetros está en progreso y ha probado varias combinaciones de C, gamma y kernel. Es un buen enfoque para encontrar los mejores parámetros el modelo, La precisión y recall para clases como drizzle, fog, y snow son bajas. Esto puede ser un indicio de que hay pocos datos para estas clases o que el modelo no está capturando bien estas categorías.
La precisión y recall son bastante buenas para la clase rain y sun.

Cada combinación de hiperparámetros se evalúa con validación cruzada (5 pliegues, según el resultado) para estimar cómo generalizaría el modelo a nuevos datos. Por ejemplo:

css
Copiar código
[CV] END ...................C=0.1, gamma=1, kernel=rbf; total time=0.0s
[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=0.0s
Mejores combinaciones de hiperparámetros: Según los resultados, algunos modelos presentan combinaciones de parámetros que lograron un ajuste rápido (indicando que la optimización fue rápida y eficiente), pero esto también puede estar relacionado con el tamaño del conjunto de datos o la complejidad del modelo.

Posibles problemas observados
Advertencias de métricas indefinidas: Varios mensajes indican que algunas clases (drizzle, fog) no se predijeron en absoluto, por lo que la métrica de precisión es indefinida para esas clases. Esto puede significar que el modelo no es lo suficientemente flexible o que los datos para esas clases son insuficientes o desbalanceados.

Desempeño del modelo:

El modelo con mayor precisión predijo correctamente la clase rain y sun, pero tuvo problemas con clases menos representadas como drizzle y fog.
El kernel rbf parece haber sido probado con múltiples valores de gamma, y algunos valores bajos de gamma y valores moderados de C tienden a mejorar la precisión.

![image](https://github.com/user-attachments/assets/d0d86611-6e51-4962-8510-3fb8cca4fdfc)

![evaluaciondemodelo](https://github.com/user-attachments/assets/63582cfb-2ee9-45df-843b-e0c2c21baf99)

La búsqueda de hiperparámetros nos ayuda a seleccionar el mejor modelo, pero también revela que los datos pueden estar desequilibrados o que algunas clases son difíciles de predecir. Para mejorar el desempeño general, debemos

Ajustar los pesos de clase para equilibrar las clases menos frecuentes.
Realizar más pruebas con valores adicionales de C y gamma.
Considerar realizar una normalización de datos o utilizar técnicas para manejar el desbalance de clases.
![image](https://github.com/user-attachments/assets/e9a9609d-f752-410a-8f77-c50cac9424a1)



