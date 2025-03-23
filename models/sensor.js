const mongoose = require('mongoose');
const Schema = mongoose.Schema;

//Definiendo modelo 1
const sensorSchema = new Schema ({

    //_id: String,
    
    client_id:{
        type: String,
        require: true,
        min: 4,
        max: 32
    },

    analog_value: {
        type: Number,
        require: true,
    },
  
    timestamp: Date
  


  
});

const db_sensors = mongoose.model('db_sensors', sensorSchema);
module.exports = db_sensors;