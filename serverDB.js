const mongoose = require('mongoose');

const username = 'user_mongo'
const password = 'dobleq3'
const dbname = 'electronica'
const string_connection = process.env.MONGODB_URI

mongoose.connect(string_connection)
.then(()=>{
    console.log('connected to database ' + dbname)
})
.catch(error=>console.log(error))