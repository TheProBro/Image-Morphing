const Files = document.getElementsByClassName("img");
const btn=document.querySelector("button");
const arr=document.getElementsByClassName("inputs")
btn.onclick=()=>{
  const data=new FormData();
  for(let j=0; j<Files.length; j++){
    const files=Files[j]
    for(let i =0; i<files.files.length; i++){
      data.append("files", files.files[i])
    }
    arr[j].src=URL.createObjectURL(files.files[0])
  }
  console.log(...data)
  fetch('http://localhost:3000/uploads', {
    method: 'POST',
    // headers: {"Content-Type": "multipart/form-data"},
    body: data,
    
  })
  .then((res)=>res.json()).then(res=>{
    // console.log(res)
    document.getElementsByClassName('res_gif')[0].innerHTML=`<img src=${res.url} width="200px">`
    console.log(document.getElementsByClassName('res_gif')[0])
  })

}