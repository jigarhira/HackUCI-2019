const socket = io.connect(`http://${document.domain}:${location.port}`)

socket.on('data', data => {
  document.getElementById('openeyes').innerHTML = data.blink_count
  document.getElementById('is-asleep').innerHTML = data.are_eyes_open == 1 ? 'no' : 'yes'
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