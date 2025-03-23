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
from sklearn.ensemble import IsolationForest   #para deteccion de anomalias
from sklearn.neighbors import LocalOutlierFactor    #para deteccion de anomalias


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
#Nota: Si timestamp es numérico, puede ser útil extraer características como hora del día, día de la semana, etc. en lugar de usarlo directamente.
X = df[["log_luz"]].values  # Suponiendo que esta es la columna de valores

# 🔹 Aplicar Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_if"] = iso_forest.fit_predict(X)  # -1 = anomalía, 1 = normal

# 🔹 Aplicar Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df["anomaly_lof"] = lof.fit_predict(X)  # -1 = anomalía, 1 = normal

# 🔹 Extraer anomalías detectadas por cada modelo
anomalies_if = df[df["anomaly_if"] == -1]
anomalies_lof = df[df["anomaly_lof"] == -1]

# 🔹 Visualización
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["analog_value"], label="Datos normales", alpha=0.7)
plt.scatter(anomalies_if["timestamp"], anomalies_if["analog_value"], color="red", marker="x", label="Anomalías IF")
plt.scatter(anomalies_lof["timestamp"], anomalies_lof["analog_value"], color="blue", marker="o", label="Anomalías LOF", alpha=0.6)
plt.xlabel("Tiempo")
plt.ylabel("Intensidad de luz")
plt.title("Comparación de Detección de Anomalías: Isolation Forest vs LOF")
plt.legend()
plt.show()

# 🔹 Comparar la cantidad de anomalías detectadas
print(f"Anomalías detectadas por Isolation Forest: {len(anomalies_if)}")
print(f"Anomalías detectadas por LOF: {len(anomalies_lof)}")

