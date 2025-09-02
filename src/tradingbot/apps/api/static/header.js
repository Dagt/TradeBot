async function loadHeader(){
  try{
    const resp = await fetch('/static/header.html');
    const html = await resp.text();
    document.body.insertAdjacentHTML('afterbegin', html);

    const path = location.pathname;
    document.querySelectorAll('header .menu > a').forEach(link => {
      if(link.getAttribute('href') === path){
        link.classList.add('active');
      }
    });

    document.querySelectorAll('header .dropdown .dropbtn').forEach(btn => {
      btn.addEventListener('click', function(e){
        e.preventDefault();
        this.parentElement.classList.toggle('open');
      });
    });
  }catch(err){
    console.error('Failed to load header:', err);
  }
}

document.addEventListener('DOMContentLoaded', loadHeader);
