import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#el archivo movies2.csv contiene distintos valores. Para realizar la regresión lineal solo necesitamos los valores numéricos.
peliculas = pd.read_csv("movies2.csv")
datos_numericos = peliculas.select_dtypes(np.number)
#algunos de éstos contienen valores NaN (No es un número), por lo que vamos a reemplazarlos por 0.
datos_numericos = peliculas.select_dtypes(np.number).fillna(0)
#En nuestro caso deseamos predecir a cuánto ascenderá el monto de ventas de cierta película
objetivo = "ventas"
#las variables independientes serian todas las demas menos ventas
independientes = datos_numericos.drop(columns=objetivo).columns
modelo = LinearRegression()
modelo.fit(X=datos_numericos[independientes], y=datos_numericos[objetivo])
peliculas["ventas_prediccion"] = modelo.predict(datos_numericos[independientes])
#En la línea anterior imprimimos las columnas ventas y ventas_predicción, solo los primeros 5 resultados.
print (peliculas[["ventas", "ventas_prediccion"]].head())


