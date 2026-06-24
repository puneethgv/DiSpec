"""Built-in live dashboard (zero-infra).

Served at GET /dashboard. Polls /stats once a second and renders live tiles +
sparklines with vanilla JS canvas — no Prometheus/Grafana/CDN required. For the full
Grafana experience, scrape /metrics and import dashboards/dispec.json (see docker-compose).
"""

DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>DiSpec</title>
<style>
  body{background:#0b0e14;color:#e6e6e6;font:14px system-ui,sans-serif;margin:0;padding:24px}
  h1{font-size:18px;margin:0 0 4px}.sub{color:#8a93a2;margin-bottom:20px}
  .tiles{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:20px}
  .tile{background:#151a23;border:1px solid #232a36;border-radius:10px;padding:14px}
  .tile .v{font-size:24px;font-weight:600}.tile .k{color:#8a93a2;font-size:12px;margin-top:4px}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .card{background:#151a23;border:1px solid #232a36;border-radius:10px;padding:14px}
  .card h2{font-size:13px;color:#8a93a2;margin:0 0 8px;font-weight:500}
  canvas{width:100%;height:160px;display:block}
</style></head><body>
<h1>DiSpec — live inference metrics</h1>
<div class="sub">polling <code>/stats</code> every 1s · for full Grafana use <code>/metrics</code> + docker-compose</div>
<div class="tiles">
  <div class="tile"><div class="v" id="tps">0</div><div class="k">throughput tok/s</div></div>
  <div class="tile"><div class="v" id="running">0</div><div class="k">running</div></div>
  <div class="tile"><div class="v" id="waiting">0</div><div class="k">waiting</div></div>
  <div class="tile"><div class="v" id="reqs">0</div><div class="k">requests total</div></div>
  <div class="tile"><div class="v" id="ttft">0</div><div class="k">avg TTFT (ms)</div></div>
  <div class="tile"><div class="v" id="tpot">0</div><div class="k">avg TPOT (ms)</div></div>
</div>
<div class="charts">
  <div class="card"><h2>Throughput (tok/s)</h2><canvas id="c_tps"></canvas></div>
  <div class="card"><h2>Queue depth</h2><canvas id="c_q"></canvas></div>
</div>
<script>
const tps=[], run=[], wait=[]; const MAX=120;
let lastTok=null, lastT=null;
function draw(cv, series, colors){
  const dpr=devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr; const x=cv.getContext('2d'); x.scale(dpr,dpr);
  x.clearRect(0,0,w,h);
  let max=1; series.forEach(s=>s.forEach(v=>{if(v>max)max=v}));
  series.forEach((s,si)=>{
    x.beginPath(); x.strokeStyle=colors[si]; x.lineWidth=2;
    s.forEach((v,i)=>{const px=w*i/(MAX-1), py=h-(v/max)*(h-10)-5; i?x.lineTo(px,py):x.moveTo(px,py)});
    x.stroke();
  });
  x.fillStyle='#8a93a2'; x.font='11px system-ui'; x.fillText(max.toFixed(0), 4, 12);
}
function push(a,v){a.push(v); if(a.length>MAX)a.shift();}
async function tick(){
  try{
    const s=await (await fetch('/stats')).json();
    const now=performance.now()/1000;
    let rate=0;
    if(lastTok!==null && now>lastT) rate=(s.tokens_total-lastTok)/(now-lastT);
    lastTok=s.tokens_total; lastT=now;
    push(tps, Math.max(rate,0)); push(run, s.running); push(wait, s.waiting);
    document.getElementById('tps').textContent=rate.toFixed(1);
    document.getElementById('running').textContent=s.running;
    document.getElementById('waiting').textContent=s.waiting;
    document.getElementById('reqs').textContent=s.requests_total;
    document.getElementById('ttft').textContent=s.ttft_avg_ms.toFixed(0);
    document.getElementById('tpot').textContent=s.tpot_avg_ms.toFixed(1);
    draw(document.getElementById('c_tps'),[tps],['#4ade80']);
    draw(document.getElementById('c_q'),[run,wait],['#60a5fa','#f59e0b']);
  }catch(e){}
}
setInterval(tick,1000); tick();
</script></body></html>
"""
