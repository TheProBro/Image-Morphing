const express=require('express')
const multer = require('multer')
const cors= require('cors')
const {spawnSync} = require('child_process')
const fs= require('fs')
const path=require('path');
const app=express();
app.use(cors());
const storage=multer.diskStorage({
    destination: function(req,file, cb){
        cb(null, __dirname+"/uploads")
    },
    filename: function(req,file, cb){
        cb(null, Date.now()+file.originalname)
    }
})
const uploads = multer({storage: storage})
  


app.post('/uploads', uploads.array("files"),(req,res)=>{
    // console.log(req.body)
    // console.log(req.files)
    const _= spawnSync('python', ['app.py'], {shell: true});
    let url="";
    const imgPath='./downloads/result.gif';
    fs.readFile(imgPath, (err, data)=>{
        // error handle
        if(err) {
            throw err;
        }
        
        // get image file extension name
        const extensionName = path.extname(imgPath);
        
        // convert image file to base64-encoded string
        const base64Image = Buffer.from(data, 'binary').toString('base64');
        
        // combine all strings
        const base64ImageStr = `data:image/${extensionName.split('.').pop()};base64,${base64Image}`;
        res.json({url: base64ImageStr})
        console.log('sent');
        const directory="./uploads";
        fs.readdir(directory, (err, files) => {
            if (err) throw err;
          
            for (const file of files) {
              fs.unlink(path.join(directory, file), (err) => {
                if (err) throw err;
              });
            }
        });
    })
})


app.listen(3000);