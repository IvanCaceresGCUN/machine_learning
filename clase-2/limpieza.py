import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
# Leer el archivo Excel
df = pd.read_csv('/Users/ivan.caceres/Documents/data/seattle-weather.csv')

# Mostrar las primeras filas del DataFrame
df = df.drop_duplicates()
print(df.dtypes)
print(df.head())
print(df.isnull().sum())
