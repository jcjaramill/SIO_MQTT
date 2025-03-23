import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # para datos sesgados o uniformes
from scipy import stats   #para transformaciones
from sklearn.model_selection import train_test_split   #para entrenar modelo
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   #metricas para evaluar modelo

df = pd.read_csv('lm393.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# 🔹 Convertir a la hora de Chile (Chile/Santiago)
df['timestamp'] = df['timestamp'].dt.tz_convert('America/Santiago')

# 🔹 Graficar datos de intensidad de luz para identificar la distribución de los datos
"""
plt.figure(figsize=(10, 6))
sns.histplot(df, kde=True, bins=30)
plt.xlabel("intensidad de luz")
plt.ylabel("Frecuencia")
plt.title("Datos de Sensor de Intensidad de Luz")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
"""

#De acuerdo al grafico anterior los datos tienden a un sesgo positivo
#escala los valores entre 0 y 1
#Aplicar Transformación para Reducir el Sesgo

#Dado que la distribución es asimétrica, podemos probar diferentes transformaciones:

#Opción 1: Transformación Logarítmica (útil para distribuciones con valores muy grandes)
#Ventaja: Comprime los valores grandes y expande los pequeños, reduciendo el sesgo.
df['log_luz'] = np.log1p(df['analog_value'])  # log(1 + x) para evitar log(0)

#Opción 2: Transformación Box-Cox (mejor si hay valores cercanos a 0)
#Ventaja: Encuentra automáticamente la mejor transformación para normalizar los datos.
df['boxcox_luz'], lambda_bc = stats.boxcox(df['analog_value'] + 1)  # +1 para evitar valores 0

#Normalizar los Datos (si es necesario)
#Si queremos que los valores estén en un rango específico (0-1) para modelos como redes neuronales:
#Útil para modelos de Machine Learning sensibles a la escala de los datos.
scaler = MinMaxScaler()  # O StandardScaler()  
df["analog_val"] = scaler.fit_transform(df[["analog_value"]])


#Volver a Graficar los Datos Transformados
#Para verificar si la transformación ayudó, podemos graficar los nuevos datos:
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histograma original
sns.histplot(df['analog_value'], bins=50, kde=True, ax=axes[0])
axes[0].set_title("Distribución Original")

# Histograma después de transformación
sns.histplot(df['log_luz'], bins=50, kde=True, ax=axes[1])
axes[1].set_title("Distribución Transformada (Log)")

#plt.show()

#Si la nueva distribución es más simétrica y con forma de campana, la transformación fue efectiva.


#Preparar datos para entrenar el modelo
#Para entrenar un modelo, necesitamos dividir los datos en variables de entrada (X) y salida (y)
#Nota: Si timestamp es numérico, puede ser útil extraer características como hora del día, día de la semana, etc. en lugar de usarlo directamente.
# Seleccionar características (X) y variable objetivo (y)
df['timestamp'] = df['timestamp'].view('int64') / 1e9  # Convertir a segundos desde la época
X = df[['timestamp']]  # colocar si hay más variables disponibles
y = df['log_luz']   #transformacion seleccionada  

# Dividir en datos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Seleccionar y Entrenar un Modelo
#Para predicción de intensidad de luz (regresión), podemos usar modelos como Regresión Lineal o Random Forest.
#Opción 1: Regresión Lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Hacer predicciones
#y_pred = modelo.predict(X_test)

#Opción 2: Random Forest (mejor para datos no lineales)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar
modelo.fit(X_train, y_train)

# Predecir
y_pred = modelo.predict(X_test)



#Evaluar el Modelo
#Para evaluar qué tan bien predice el modelo, usamos métricas como:
#Valores más bajos de MAE y MSE indican mejor precisión.
#📌 Un R² cercano a 1 significa que el modelo es bueno para predecir.
print("MAE:", mean_absolute_error(y_test, y_pred))  # Error absoluto medio
print("MSE:", mean_squared_error(y_test, y_pred))  # Error cuadrático medio
print("R² Score:", r2_score(y_test, y_pred))  # Qué tan bien ajusta el modelo


#Visualizar Resultados
#Para ver cómo se comparan las predicciones con los valores reales:
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Valor Real", linestyle="dashed")
plt.plot(y_pred, label="Predicción", alpha=0.7)
plt.legend()
plt.xlabel("Tiempo")
plt.ylabel("Intensidad de Luz")
plt.title("Comparación entre Valores Reales y Predichos")
plt.show()



# 🔹 Calcular y mostrar la media de la intensidad de luz
#print(f"Estadisticas de intensidad de luz: {df['analog_value'].describe()}")
#print(df.info())