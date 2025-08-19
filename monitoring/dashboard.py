from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TradeBot Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>tailwind.config = {darkMode: 'class'};</script>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 h-screen flex flex-col">
  <header class="flex items-center justify-between bg-gray-800 p-4 shadow">
    <h1 class="text-xl font-bold">TradeBot Dashboard</h1>
    <button id="menu-btn" class="md:hidden">☰</button>
  </header>
  <div class="flex flex-1 overflow-hidden">
    <aside id="side-menu" class="bg-gray-800 w-64 p-4 space-y-2 hidden md:block">
      <nav class="space-y-2">
        <a href="#metrics" class="block p-2 rounded hover:bg-gray-700">Metrics</a>
        <a href="#configuration" class="block p-2 rounded hover:bg-gray-700">Configuration</a>
        <a href="#logs" class="block p-2 rounded hover:bg-gray-700">Logs</a>
      </nav>
    </aside>
    <main class="flex-1 overflow-y-auto p-4 space-y-8">
      <section id="metrics">
        <h2 class="text-lg font-semibold mb-4">Metrics</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div id="pnl" class="bg-gray-800 rounded p-2 h-64"></div>
          <div id="slippage" class="bg-gray-800 rounded p-2 h-64"></div>
          <div id="latency" class="bg-gray-800 rounded p-2 h-64"></div>
        </div>
      </section>
      <section id="configuration">
        <h2 class="text-lg font-semibold mb-4">Strategies</h2>
        <div class="bg-gray-800 rounded p-4 overflow-x-auto">
          <table class="min-w-full text-sm">
            <thead class="bg-gray-700">
              <tr><th class="p-2 text-left">Name</th><th class="p-2 text-left">Status</th><th class="p-2 text-left">Actions</th></tr>
            </thead>
            <tbody id="strategy-body"></tbody>
          </table>
        </div>
      </section>
      <section id="logs">
        <h2 class="text-lg font-semibold mb-4">Logs</h2>
        <pre id="log-output" class="bg-gray-800 rounded p-4 h-64 overflow-y-auto text-xs"></pre>
      </section>
    </main>
  </div>
  <script>
  const menuBtn = document.getElementById('menu-btn');
  const sideMenu = document.getElementById('side-menu');
  menuBtn?.addEventListener('click', () => sideMenu.classList.toggle('hidden'));

  const pnlTrace = {x: [], y: [], mode: 'lines', name: 'PnL'};
  const slippageTrace = {x: [], y: [], mode: 'lines', name: 'Slippage'};
  const latencyTrace = {x: [], y: [], mode: 'lines', name: 'Order Latency'};

  Plotly.newPlot('pnl', [pnlTrace], {title:'PnL', paper_bgcolor:'#1f2937', plot_bgcolor:'#1f2937', font:{color:'#e5e7eb'}});
  Plotly.newPlot('slippage', [slippageTrace], {title:'Slippage (bps)', paper_bgcolor:'#1f2937', plot_bgcolor:'#1f2937', font:{color:'#e5e7eb'}});
  Plotly.newPlot('latency', [latencyTrace], {title:'Order Latency (s)', paper_bgcolor:'#1f2937', plot_bgcolor:'#1f2937', font:{color:'#e5e7eb'}});

  async function updateMetrics(){
    try {
      const now = new Date();
      const metrics = await fetch('/metrics').then(r => r.json());
      pnlTrace.x.push(now);
      pnlTrace.y.push(metrics.pnl || 0);
      slippageTrace.x.push(now);
      slippageTrace.y.push(metrics.avg_slippage_bps || 0);
      latencyTrace.x.push(now);
      latencyTrace.y.push(metrics.avg_order_latency_seconds || 0);
      Plotly.update('pnl', {x:[pnlTrace.x], y:[pnlTrace.y]});
      Plotly.update('slippage', {x:[slippageTrace.x], y:[slippageTrace.y]});
      Plotly.update('latency', {x:[latencyTrace.x], y:[latencyTrace.y]});
    } catch(err) {
      console.error('Error fetching metrics', err);
    }
  }

  async function updateStrategies(){
    try {
      const data = await fetch('/strategies/status').then(r => r.json());
      const tbody = document.getElementById('strategy-body');
      tbody.innerHTML = '';
      for (const [name, status] of Object.entries(data.strategies || {})) {
        const row = document.createElement('tr');
        row.innerHTML = `<td class="p-2">${name}</td><td class="p-2">${status}</td>` +
          `<td class="p-2"><button class="bg-red-600 px-2 py-1 rounded mr-2" onclick="controlStrategy('${name}','disable')">Pause</button>` +
          `<button class="bg-green-600 px-2 py-1 rounded" onclick="controlStrategy('${name}','enable')">Resume</button></td>`;
        tbody.appendChild(row);
      }
    } catch(err) {
      console.error('Error fetching strategies', err);
    }
  }

  async function controlStrategy(name, action){
    await fetch(`/strategies/${name}/${action}`, {method: 'POST'});
    updateStrategies();
  }

  async function updateLogs(){
    try {
      const logs = await fetch('/logs').then(r => r.text());
      document.getElementById('log-output').textContent = logs;
    } catch(err) {
      console.error('Error fetching logs', err);
    }
  }

  setInterval(() => {updateMetrics(); updateStrategies(); updateLogs();}, 5000);
  updateMetrics();
  updateStrategies();
  updateLogs();
  </script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the Plotly dashboard with live metrics and controls."""
    return HTMLResponse(content=_HTML)

