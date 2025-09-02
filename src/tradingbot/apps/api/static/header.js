async function loadHeader(active){
  try {
    const resp = await fetch('/static/header.html');
    const html = await resp.text();
    document.getElementById('header').innerHTML = html;
    if (active) {
      const link = document.querySelector(`.menu a[data-page="${active}"]`);
      if (link) link.classList.add('active');
    }
  } catch (e) {
    console.error('Failed to load header:', e);
  }
}
