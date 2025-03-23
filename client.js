const mqtt = require('mqtt');

const client_id = 'client-001';
const options = {
  client: client_id,
  clean: true
};

const brokerUrl = 'mqtt://192.168.1.82';
const client = mqtt.connect(brokerUrl, options);

function EventoPacketSend(packet) {
  console.log('El paquete enviado es:', packet.cmd);
}

function EventoPacketReceive(packet) {
  console.log('El paquete recibido es:', packet.cmd);
  if (packet.cmd === 'pingresp') {
    client.subscribe('topic/04');
  }
}

function EventoConectar() {
  client.subscribe('topic/03', function (err) {
    if (!err) {
      console.log('Cliente mqtt-sio conectado');
    }
  });
}

function EventoMensaje(topic, message) {
  if (topic === 'topic/03') {
    const string = message.toString();
    const json = JSON.parse(string);
    console.log(json);
    if (json.speed === 35) {
      client.unsubscribe('topic/03');
    }
  }

}

client.on('connect', EventoConectar);
client.on('message', EventoMensaje);
client.on('packetreceive', EventoPacketReceive);
client.on('packetsend', EventoPacketSend);

// Bucle para publicar mensajes periódicamente
setInterval(() => {
  const mensaje = {
    client: client_id,
    speed: Math.floor(Math.random() * (200 - 180 + 1) + 180)
  };
  client.publish('topic/03', JSON.stringify(mensaje));
  console.log(`Publicado en topic/04: ${JSON.stringify(mensaje)}`);
}, 1000); // Publicar cada 2 segundos (puedes ajustar el tiempo según tus necesidades)
