import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib   #para desplegar modelo

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # para datos sesgados o uniformes
from scipy import stats   #para transformaciones
from sklearn.model_selection import train_test_split   #para entrenar modelo
from sklearn.preprocessing import LabelEncoder   #convierte etiquetas en numeros
from sklearn.tree import DecisionTreeClassifier   #modelo
from sklearn.ensemble import RandomForestClassifier   #modelo
from sklearn.metrics import classification_report, confusion_matrix   #para evaluar que tan bien clasifica el modelo
#from imblearn.over_sampling import SMOTE # para balanceo




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

plt.show()

#Si la nueva distribución es más simétrica y con forma de campana, la transformación fue efectiva.


#Seleccionar y Entrenar un Modelo
#Entrenamiento de un Modelo de Clasificación para Sensor de Luz
#Ahora cambiaremos el enfoque:
#📌 En lugar de predecir valores numéricos de luz, clasificaremos las condiciones de iluminación en "Baja", "Media" y "Alta".
# Definir los umbrales para la clasificación
def clasificar_luz(valor):
    if valor == 0:
        return "Luz solar intensa"
    elif 0 < valor <= 500:
        return "Reflejos con sombra"
    elif 500 < valor <= 2000:
        return "sombra"
    else:
        return "oscuro"

# Aplicar la clasificación a los datos
df['categoria_luz'] = df['analog_value'].apply(clasificar_luz)

# Verificar la distribución de clases
print(df['categoria_luz'].value_counts())

#Preparar los Datos para el Modelo
#Ahora convertimos los datos en una estructura adecuada para entrenamiento.
# Seleccionar características de entrada (X) y etiquetas de salida (y)
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / (df['timestamp'].max() - df['timestamp'].min())
#X = df[['timestamp', 'temperatura', 'humedad']]  # Variables relevantes
X = df[['timestamp', 'analog_value']]
y = df['categoria_luz']

# Convertir etiquetas en números (Baja = 0, Media = 1, Alta = 2)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Dividir en datos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar las dimensiones
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)

#Seleccionar y Entrenar un Modelo
#Para clasificación, podemos usar árboles de decisión, Random Forest o SVM.
#🔹 Opción 1: Árbol de Decisión
# Crear y entrenar el modelo
modelo1 = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo1.fit(X_train, y_train)

# Hacer predicciones
#y_pred = modelo1.predict(X_test)

#🔹 Opción 2: Random Forest (mejor rendimiento)
# Crear el modelo
modelo2 = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar
modelo2.fit(X_train, y_train)

# Predecir
y_pred = modelo2.predict(X_test)


#Evaluar el Modelo
#Para evaluar qué tan bien clasifica el modelo, usamos matriz de confusión y métricas de precisión.
# Matriz de confusión
#🔹 Precisión alta (>90%) → Buen modelo.
#🔹 Precisión baja (<70%) → Se pueden ajustar hiperparámetros o usar otro modelo.
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


#Visualizar Resultados
#Para ver cómo predice el modelo en comparación con los valores reales:
# Matriz de confusión visual
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()


#joblib.dump(modelo2, 'modelo.pkl')

#print("✅ Modelo guardado como 'modelo.pkl'")


