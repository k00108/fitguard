"""Dependency-free local Web dashboard for the FitGuard live runtime."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FitGuard 双模态实时监测</title>
<style>
body{margin:0;background:#080b10;color:#eef;font-family:Arial,"Microsoft YaHei",sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:#111722}
h1{font-size:20px;margin:0}.wrap{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:14px;padding:14px}
.card{background:#111722;border:1px solid #273244;border-radius:10px;padding:12px}
#frame{width:100%;max-height:calc(100vh - 105px);object-fit:contain;background:#000}
.state{font-size:27px;font-weight:700;margin:8px 0}.ok{color:#37d67a}.warn{color:#ffd166}.bad{color:#ff5c70}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.metric{background:#0a0f17;padding:9px;border-radius:7px}
.label{font-size:12px;color:#93a4bb}.value{font-size:18px;margin-top:3px}button{padding:8px 14px;cursor:pointer}
@media(max-width:850px){.wrap{grid-template-columns:1fr}#frame{max-height:none}}
</style></head><body>
<header><h1>FitGuard 无线腰部双模态实时监测</h1><button onclick="document.documentElement.requestFullscreen()">全屏</button></header>
<main class="wrap"><section class="card"><img id="frame" src="/frame.jpg"></section>
<aside class="card"><div class="label">最终状态</div><div id="state" class="state warn">等待数据</div>
<div class="grid">
<div class="metric"><div class="label">Stable</div><div id="p0" class="value">--</div></div>
<div class="metric"><div class="label">Mild</div><div id="p1" class="value">--</div></div>
<div class="metric"><div class="label">Marked</div><div id="p2" class="value">--</div></div>
<div class="metric"><div class="label">传感器状态</div><div id="sensor" class="value">--</div></div>
<div class="metric"><div class="label">人物跟踪质量</div><div id="quality" class="value">--</div></div>
<div class="metric"><div class="label">CNN推理</div><div id="latency" class="value">--</div></div>
<div class="metric"><div class="label">运动指标</div><div id="motion" class="value">--</div></div>
<div class="metric"><div class="label">重度异常事件</div><div id="event" class="value">未触发</div></div>
</div><p class="label" id="backend"></p><p class="label">该原型用于运动稳定性研究，不用于临床诊断。</p></aside></main>
<script>
async function refresh(){
 try{let s=await (await fetch('/status.json?'+Date.now())).json();
 let st=document.getElementById('state'); st.textContent=s.display_state||s.sensor_state||'等待数据';
 st.className='state '+(s.display_state==='Marked Abnormality'?'bad':(s.sensor_state==='OK'?'ok':'warn'));
 let p=s.probabilities||[]; for(let i=0;i<3;i++)document.getElementById('p'+i).textContent=p[i]==null?'--':(100*p[i]).toFixed(1)+'%';
 document.getElementById('sensor').textContent=s.sensor_state||'--';
 document.getElementById('quality').textContent=s.vision_quality==null?'--':Number(s.vision_quality).toFixed(2);
 document.getElementById('latency').textContent=s.inference_ms==null?'--':Number(s.inference_ms).toFixed(1)+' ms';
 let m=s.motion_indicators||{}; document.getElementById('motion').textContent=m.summary||'--';
 let e=s.event||{}; document.getElementById('event').textContent=e.active?(e.event_id||'记录中'):'未触发';
 document.getElementById('backend').textContent='推理后端：'+(s.backend||'--');
 document.getElementById('frame').src='/frame.jpg?'+Date.now();
 }catch(e){} setTimeout(refresh,350)} refresh();
</script></body></html>"""


class FitGuardDashboardServer:
    def __init__(self, port=8088, host="0.0.0.0"):
        self.port = int(port)
        self.host = host
        self._lock = threading.Lock()
        self._jpeg = b""
        self._status = {"sensor_state": "STARTING"}
        self._server = None
        self._thread = None
        self.error = None

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                with owner._lock:
                    jpeg = owner._jpeg
                    status = dict(owner._status)
                if path == "/" or path == "/index.html":
                    body = _HTML.encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif path == "/frame.jpg":
                    body = jpeg
                    content_type = "image/jpeg"
                elif path == "/status.json":
                    body = json.dumps(status, ensure_ascii=False).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
            self._thread = threading.Thread(
                target=self._server.serve_forever, name="fitguard-dashboard", daemon=True
            )
            self._thread.start()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._server = None
        return self.status()

    def update(self, jpeg=None, status=None):
        with self._lock:
            if jpeg:
                self._jpeg = bytes(jpeg)
            if status is not None:
                self._status = dict(status)
                self._status["dashboard_updated_monotonic"] = time.monotonic()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def status(self):
        return {
            "running": self._server is not None,
            "host": self.host,
            "port": self.port,
            "url_hint": f"http://<PYNQ-IP>:{self.port}/",
            "error": self.error,
        }

