import requests
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 URL del servidor que proporciona los datos (modifica según tu caso)
URL = "http://localhost:3200/api/sensors/esp32_lm393"

# 🔹 Hacer la solicitud GET al servidor
response = requests.get(URL)

# 🔹 Verificar si la solicitud fue exitosa
if response.status_code == 200:
    data = response.json()  # Convertir la respuesta JSON a diccionario
    df = pd.DataFrame(data)  # Convertir a DataFrame

    # Guardar el DataFrame como archivo CSV
    df.to_csv('lm393.csv', index=False)
    
    # 🔹 Convertir a datetime y asignar UTC si los datos están sin zona horaria
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 🔹 Convertir a la hora de Chile (Chile/Santiago)
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/Santiago')

    # 🔹 Graficar datos de intensidad de luz
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['analog_value'], label="Intensidad de luz", color="orange")
    plt.xlabel("Tiempo")
    plt.ylabel("Intensidad de Luz")
    plt.title("Datos de Sensor de Intensidad de Luz")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    # 🔹 Calcular y mostrar la media de la intensidad de luz
    print(f"Promedio de intensidad de luz: {df['analog_value'].mean()}")
    print(df.info())

else:
    print(f"Error al obtener datos. Código de estado: {response.status_code}")
