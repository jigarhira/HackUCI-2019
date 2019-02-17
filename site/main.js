const socket = io.connect(`http://${document.domain}:${location.port}`)

socket.on('openEyes', data => {
    console.log('openEyes', data)
})
document.getElementById("openeyes").innerHTML = data;
openeyes=data;

socket.on('closeEyes', data => {
    console.log('closeEyes', data)
})

socket.on('sleep', data => {
    console.log('sleep', data)
})

socket.on('smsSent', data => {
    console.log('smsSent', data)
})

// If possible add to this code so UI is nice and functional

$(document).ready(function() {

  function toggleSidebar() {
    $(".button").toggleClass("active");
    $("main").toggleClass("move-to-left");
    $(".sidebar-item").toggleClass("active");
  }

  $(".button").on("click tap", function() {
    toggleSidebar();
  });

  $(document).keyup(function(e) {
    if (e.keyCode === 27) {
      toggleSidebar();
    }
  });

});