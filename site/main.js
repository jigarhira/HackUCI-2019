const socket = io.connect(`http://${document.domain}:${location.port}`)

socket.on('openEyes', data => {
    console.log('openEyes', data)
})

socket.on('closeEyes', data => {
    console.log('closeEyes', data)
})

socket.on('sleep', data => {
    console.log('sleep', data)
})

socket.on('smsSent', data => {
    console.log('smsSent', data)
})