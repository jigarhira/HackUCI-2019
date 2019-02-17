const socket = io.connect(`http://${document.domain}:${location.port}`)

// If possible add to this code so UI is nice and functional
socket.on('data', data => {
  console.log(data)
  if (data.are_eyes_open != 1) {
    console.log('eyes are not open!')
  }
  console.log('blinks: ' + data.blink_count)
})

$(document).ready(function() {
  setInterval(() => {
    socket.emit('data', null)
  }, 1000)
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
