const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const WebSocket = require('ws');
const mqtt = require('mqtt');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config();
require('./serverDB');

const app = express();
const port = process.env.PORT || 3200;

// Middleware
app.use(cors());
app.use(express.urlencoded({ extended: false }));
app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  next();
});

// Static files
app.set(__dirname);
app.use(express.static(path.resolve(__dirname, './public/')));

// Definir el esquema y modelo de Mongoose
const sensorDataSchema = new mongoose.Schema({
  client_id: { type: String, required: true },
  analog_value: { type: Number, required: true },
  timestamp: { type: Date, default: Date.now },
});

// Configuración para eliminar automáticamente `__v` y `_id` si es necesario
sensorDataSchema.set('toJSON', {
  transform: (doc, ret) => {
    delete ret.__v;
    return ret;
  },
});

const db_sensors = mongoose.model('db_sensors', sensorDataSchema);


// Definir el esquema y modelo de Mongoose
const sensorDataSchema1 = new mongoose.Schema({
  client_id: { type: String, required: true },
  analog_value: { type: Number, required: true },
  timestamp: { type: Date, default: Date.now },
});

// Configuración para eliminar automáticamente `__v` y `_id` si es necesario
sensorDataSchema1.set('toJSON', {
  transform: (doc, ret) => {
    delete ret.__v;
    return ret;
  },
});

const db_sensores = mongoose.model('db_sensores', sensorDataSchema1);




// Rutas para datos históricos de sensores
app.get('/', (req, res) => {
  res.json({
    status: 'success',
    message: 'WELCOME API SENSOR MONITORING',
  });
});


app.get('/api/sensors', async (req, res) => {
  const { startDate, endDate } = req.query;

  try {
    // Convertir los valores de startDate y endDate a objetos Date si existen
    const start = startDate ? new Date(parseInt(startDate, 10)) : null;
    const end = endDate ? new Date(parseInt(endDate, 10)) : null;

    if ((startDate && isNaN(start)) || (endDate && isNaN(end))) {
      return res.status(400).json({
        error: 'Invalid startDate or endDate. Please provide valid timestamps in milliseconds.',
      });
    }

    // Construir la consulta
    const query = {};

    if (start || end) {
      query.timestamp = {};
      if (start) query.timestamp.$gte = start;
      if (end) query.timestamp.$lte = end;
    }

    // Obtener los datos de MongoDB
    const data = await db_sensors.find(query).sort({ timestamp: 1 });

    // Formatear los datos para Grafana
    const formattedData = data.map((item) => ({
      timestamp: item.timestamp.toISOString(), // Formato ISO 8601
      analog_value: item.analog_value,          // Valor del sensor
      client_id: item.client_id,         // ID del cliente
    }));

    // Enviar los datos formateados
    res.status(200).json(formattedData);
  } catch (err) {
    console.error('Error al obtener datos:', err);
    res.status(500).send('Error interno del servidor');
  }
});


app.get('/api/sensors/list', async (req, res) => {
  try {
    const sensors = await db_sensors.distinct('client_id'); // Obtener sensores únicos
    res.status(200).json(sensors);
  } catch (err) {
    console.error('Error al obtener sensores:', err);
    res.status(500).send('Error interno del servidor');
  }
});

app.get('/api/sensors/:client_id', async (req, res) => {
  const { client_id } = req.params;
  const { startDate, endDate } = req.query;
  console.log(req.query);

  try {
    // Convertir los valores de timestamp a Date si existen
    const start = startDate ? new Date(parseInt(startDate, 10)) : null;
    const end = endDate ? new Date(parseInt(endDate, 10)) : null;

    if ((startDate && isNaN(start)) || (endDate && isNaN(end))) {
      return res.status(400).json({
        error: 'Invalid startDate or endDate. Please provide valid timestamps in milliseconds.',
      });
    }

    const query = { client_id };

    // Agregar filtro de fechas solo si son válidas
    if (start || end) {
      query.timestamp = {};
      if (start) query.timestamp.$gte = start;
      if (end) query.timestamp.$lte = end;
    }

    const data = await db_sensors.find(query).sort({ timestamp: 1 });
    return res.status(200).json(data)



  } catch (err) {
    console.error('Error al obtener datos históricos:', err);
    res.status(500).send('Error interno del servidor');
  }
});


// Crear servidor HTTP y WebSocket
const server = app.listen(port, () => {
  console.log(`Servidor Express en ejecución en el puerto ${port}`);
});

// Configuración de WebSocket
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  console.log('Cliente WebSocket conectado');

  // Simulación de datos de tres sensores con rangos específicos
  const sensorConfig = {
    T1: { min: 18, max: 21 },
    T2: { min: 32, max: 35 },
    T3: { min: 25, max: 27 },
  };

  const intervalId = setInterval(async () => {
    for (const [client_id, range] of Object.entries(sensorConfig)) {
      const analog_value = Math.random() * (range.max - range.min) + range.min; // Generar valor dentro del rango
      const timestamp = new Date();

      // Guardar datos en MongoDB
      //const dataPoint = new db_sensors({ client_id, analog_value, timestamp });
      //await dataPoint.save();

      // Enviar datos al cliente

      const data = { client_id, analog_value, timestamp };
      //ws.send(JSON.stringify(data));
    }
  }, 1000);

  // Manejar cierre de la conexión
  ws.on('close', () => {
    console.log('Cliente WebSocket desconectado');
    clearInterval(intervalId);
  });
});

// Configuración del cliente MQTT
const mqttClient = mqtt.connect('mqtt://192.168.1.84:1883');

mqttClient.on('connect', () => {
  console.log('Cliente MQTT conectado al broker');
  mqttClient.subscribe('topic/03', (err) => {
    if (err) {
      console.error('Error al suscribirse al tema MQTT:', err);
    } else {
      console.log('Suscrito al tema: topic/03');
    }
  });
});

mqttClient.on('message', async (topic, message) => {
  try {
    if (topic === 'topic/03') {
      const json = JSON.parse(message.toString());
      const { client_id, analog_value } = json;
      const timestamp = new Date();

      // Guardar datos en MongoDB
      const dataPoint = new db_sensors({ client_id, analog_value, timestamp });
      await dataPoint.save();

      // Hacer la predicción con Python
      const pythonProcess = spawn("python", ["predict.py"]);

      let result = "";
      let errorResult = "";

      const dataString = JSON.stringify(
        [{
         timestamp: timestamp.getTime(), 
         analog_value 
        }
      ]);

      pythonProcess.stdin.write(dataString);
      pythonProcess.stdin.end();

      pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        errorResult += data.toString();
      });

      pythonProcess.on("close", (code) => {
        if (errorResult) {
          console.error("❌ Error en Python:", errorResult);
          return;
        }

        if (!result) {
          console.error("❌ Python no devolvió datos");
          return;
        }

        try {
          const prediction = JSON.parse(result);
          console.log("✅ Predicción recibida:", prediction);

          let predict = prediction.prediction[0]

          if (predict === 0) { 
            predict = 'luz solar intensa'
          }

          if (predict === 1) { 
            predict = 'reflejos de sol y sombra'
          }

          if (predict === 2) { 
            predict = 'oscuridad'
          }

          if (predict === 3) { 
              predict = 'sombra'
          }

          // Enviar datos a través del WebSocket con la predicción
          wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({ client_id, analog_value, timestamp, predict }));
              console.log(analog_value)
            }
          });

        } catch (error) {
          console.error("❌ Error al parsear JSON:", error);
        }
      });
    }
  } catch (err) {
    console.error('Error al procesar el mensaje MQTT:', err);
  }
});


mqttClient.on('error', (err) => {
  console.error('Error en el cliente MQTT:', err);
});
