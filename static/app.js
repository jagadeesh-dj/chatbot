input=document.getElementById('msg').focus()

function speech(){
    input=document.getElementById('msg')
    bot=document.getElementById('bot')
    chatbot=document.getElementById('chatbot')


    fetch('/speech/',{
        method:'GET',
    })
    .then(res=>res.json())
    .then(data=>{
        console.log(data)
        let trs=document.createElement('tr')
        let tds=document.createElement('td')
        let small=document.createElement('small')
        trs.appendChild(tds)
        trs.appendChild(small)
        chatbot.appendChild(trs)
        tds.textContent=data.text
        let date=new Date()
        small.innerHTML=date.getHours()+":"+date.getMinutes()
        tds.style.backgroundColor="lightblue"
        tds.style.width='fit-content'
        tds.style.padding="1rem 1rem"
        trs.style.float="right"
        trs.style.display='flex'
        small.style.marginLeft="10px"
        tds.style.borderRadius="10px"
        scroll()
        let tr=document.createElement('tr')
        let td=document.createElement('td')
        tr.appendChild(td)
        chatbot.appendChild(tr)
        datas=data.res
        td.textContent=datas
        td.style.backgroundColor="lightgreen"
        td.style.width='fit-content'
        td.style.padding="1rem 1rem"
        td.style.borderRadius="10px"
        td.style.float="left"
        td.style.marginTop="4px"
        td.style.marginBottom="4px"

        scroll()

    })
    }


function typingdata(){
    input=document.getElementById('msg')
    bot=document.getElementById('bot')
    chatbot=document.getElementById('chatbot')

    if(input.value === ""){
        return
    }
    else{

    
    let input_val=input.value
    let tr=document.createElement('tr')
    let td=document.createElement('td')
    let small=document.createElement('small')
    tr.appendChild(td)
    tr.appendChild(small)
    chatbot.appendChild(tr)
    td.textContent=input_val
    let date=new Date()
    small.innerHTML=date.getHours()+":"+date.getMinutes()
    td.style.backgroundColor="lightblue"
    td.style.width='fit-content'
    td.style.padding="1rem 1rem"
    tr.style.float="right"
    tr.style.display='flex'
    small.style.marginLeft="10px"
    td.style.borderRadius="10px"
    scroll()
    fetch('/chatbot/'+input_val+'/',{
        method:'GET',
    })
    .then(res=>res.json())
    .then(data=>{
        let tr=document.createElement('tr')
        let td=document.createElement('td')
        tr.appendChild(td)
        chatbot.appendChild(tr)
        let datas=data.res
        td.textContent=datas
        td.style.backgroundColor="lightgreen"
        td.style.width='fit-content'
        td.style.padding="1rem 1rem"
        td.style.borderRadius="10px"
        td.style.float="left"
        td.style.marginTop="4px"
        td.style.marginBottom="4px"
        scroll()

    })
    }
    input.value=""
}
document.addEventListener("DOMContentLoaded",(e)=>{
    btn=document.querySelector('#btn')
    btn.addEventListener('click',(e)=>{
    e.preventDefault()
    console.log('typing')
    typingdata()
    })
});
document.addEventListener("DOMContentLoaded",(e)=>{
    btn2=document.querySelector('#btn2')
    btn2.addEventListener('click',(e)=>{
    e.preventDefault()
    console.log('speech')
    speech()
    })
});

function scroll(){

    const scro=document.getElementById('messages')
    scro.scrollTop=scro.scrollHeight
}

// const recognitio=new webKitSpeechRecognition() || new SpeechRecognition();
