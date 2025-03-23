const mongoose = require('mongoose');

const username = 'user_mongo'
const password = 'dobleq3'
const dbname = 'electronica'
const string_connection = `mongodb+srv://${username}:${password}@cluster0.sj5du.mongodb.net/${dbname}?retryWrites=true&w=majority`

mongoose.connect(string_connection)
.then(()=>{
    console.log('connected to database ' + dbname)
})
.catch(error=>console.log(error))