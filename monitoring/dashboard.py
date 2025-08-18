from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>TradeBot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
<h1>TradeBot Metrics Dashboard</h1>
<div id="pnl" style="width:100%;height:300px;"></div>
<div id="slippage" style="width:100%;height:300px;"></div>
<div id="latency" style="width:100%;height:300px;"></div>
<h2>Strategies</h2>
<table id="strategies" border="1" cellspacing="0" cellpadding="5">
  <thead><tr><th>Name</th><th>Status</th><th>Actions</th></tr></thead>
  <tbody id="strategy-body"></tbody>
</table>
<script>
const pnlTrace = {x: [], y: [], mode: 'lines', name: 'PnL'};
const slippageTrace = {x: [], y: [], mode: 'lines', name: 'Slippage'};
const latencyTrace = {x: [], y: [], mode: 'lines', name: 'Order Latency'};

Plotly.newPlot('pnl', [pnlTrace], {title: 'PnL'});
Plotly.newPlot('slippage', [slippageTrace], {title: 'Slippage (bps)'});
Plotly.newPlot('latency', [latencyTrace], {title: 'Order Latency (s)'});

async function updateMetrics(){
  try {
    const now = new Date();
    const pnl = await fetch('/metrics/pnl').then(r => r.json());
    const slip = await fetch('/metrics/slippage').then(r => r.json());
    const lat = await fetch('/metrics/latency').then(r => r.json());

    pnlTrace.x.push(now);
    pnlTrace.y.push(pnl.pnl || 0);
    slippageTrace.x.push(now);
    slippageTrace.y.push(slip.avg_slippage_bps || 0);
    latencyTrace.x.push(now);
    latencyTrace.y.push(lat.avg_order_latency_seconds || 0);

    Plotly.update('pnl', {x: [pnlTrace.x], y: [pnlTrace.y]});
    Plotly.update('slippage', {x: [slippageTrace.x], y: [slippageTrace.y]});
    Plotly.update('latency', {x: [latencyTrace.x], y: [latencyTrace.y]});
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
      row.innerHTML = `<td>${name}</td><td>${status}</td>` +
        `<td><button onclick="controlStrategy('${name}','disable')">Pause</button>` +
        `<button onclick="controlStrategy('${name}','enable')">Resume</button></td>`;
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

setInterval(() => {updateMetrics(); updateStrategies();}, 5000);
updateMetrics();
updateStrategies();
</script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the Plotly dashboard with live metrics and controls."""
    return HTMLResponse(content=_HTML)
