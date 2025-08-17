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
<div id="pnl" style="width:100%;height:400px;"></div>
<div id="fills" style="width:100%;height:400px;"></div>
<script>
const pnlTrace = {x: [], y: [], mode: 'lines', name: 'PnL'};
const fillsTrace = {x: [], y: [], mode: 'lines', name: 'Fills'};
Plotly.newPlot('pnl', [pnlTrace], {title: 'PnL'});
Plotly.newPlot('fills', [fillsTrace], {title: 'Fills'});
async function fetchData(){
  try {
    const resp = await fetch('/metrics/summary');
    const data = await resp.json();
    const now = new Date();
    pnlTrace.x.push(now);
    pnlTrace.y.push(data.pnl || 0);
    fillsTrace.x.push(now);
    fillsTrace.y.push(data.fills || 0);
    Plotly.update('pnl', {x: [pnlTrace.x], y: [pnlTrace.y]});
    Plotly.update('fills', {x: [fillsTrace.x], y: [fillsTrace.y]});
  } catch(err) {
    console.error('Error fetching metrics', err);
  }
}
setInterval(fetchData, 5000);
fetchData();
</script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve a simple Plotly dashboard polling /metrics/summary."""
    return HTMLResponse(content=_HTML)
