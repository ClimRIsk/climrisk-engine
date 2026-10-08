"""Extension entry point: a chat command and the confirmation page for authenticated pushes (loopback only)."""
from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import push

PAGE = """<!doctype html><meta charset="utf-8"><title>Maintainer push</title>
<style>body{font:16px/1.5 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1b2430}pre{background:#eef2f7;padding:8px;overflow:auto}.bad{color:#b00020}.ok{color:#0a7a3b}button{font:inherit;padding:.5rem 1rem}input{font:inherit;padding:.4rem}</style>
<h2>Push to GitHub (private maintainer tool)</h2><div id="out">Checking what would be pushed…</div>
<script>
var h=(location.hash.match(/key=([^&]+)/)||[])[1];var K=h?decodeURIComponent(h):(localStorage.getItem('cri_api_key')||'');
function api(p,b){var H={'Content-Type':'application/json'};if(K)H['X-API-Key']=K;return fetch(p,{method:'POST',headers:H,body:JSON.stringify(b||{})}).then(r=>r.json().then(j=>({ok:r.ok,j:j})));}
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
var tok=null;
api('/maintainer/propose').then(function(r){var j=r.j,o=document.getElementById('out');
 if(!r.ok||j.ok===false){o.innerHTML='<p class=bad>'+esc(j.error||JSON.stringify(j.detail||j))+'</p>';return;}
 tok=j.token;var s='';(j.plans||[]).forEach(function(p){s+='<h3>'+esc(p.name)+'</h3>';
  if(!p.ok){s+='<p class=bad>'+esc(p.why)+'</p>';return;}
  s+='<p>Branch <b>'+esc(p.branch)+'</b>, <b>'+p.ahead+'</b> commit(s) ahead of '+esc(p.base)+(p.uncommitted?' ('+p.uncommitted+' uncommitted file(s) will NOT be pushed)':'')+'</p>';
  if(p.commits.length)s+='<pre>'+esc(p.commits.join('\\n'))+'\\n'+esc(p.summary)+'</pre>';
  if(p.secrets.length)s+='<p class=bad>Blocked by the secret scan:<br>'+p.secrets.map(esc).join('<br>')+'</p>';});
 s+='<p>'+esc(j.note)+'</p>';
 if(tok)s+='<p>PIN: <input id=pin type=password autocomplete=off> <button onclick=go()>Push now</button></p><p id=res></p>';
 o.innerHTML=s;});
function go(){var pin=document.getElementById('pin').value;api('/maintainer/execute',{token:tok,pin:pin}).then(function(r){var j=r.j;
 document.getElementById('res').innerHTML=(j.ok?'<span class=ok>Pushed.</span>':'<span class=bad>'+esc(j.error||'Not pushed')+'</span>')+'<pre>'+esc((j.results||[]).map(x=>x.repo+': '+(x.ok?'ok':'FAILED')+'\\n'+(x.output||x.error||'')).join('\\n\\n'))+'</pre>';tok=null;});}
</script>"""


class Exec(BaseModel):
    token: str
    pin: str


def _local(request: Request) -> bool:
    return (request.client.host if request.client else "") in ("127.0.0.1", "::1", "localhost")


def make_router(auth_dep) -> list:
    prot = APIRouter(prefix="/maintainer", tags=["Maintainer (private)"], dependencies=[auth_dep])
    pub = APIRouter(prefix="/maintainer", tags=["Maintainer (private)"])

    @prot.get("/status")
    def st(request: Request) -> dict:
        if not _local(request):
            raise HTTPException(403, "local only")
        return push.status()

    @prot.post("/propose")
    def prop(request: Request) -> dict:
        if not _local(request):
            raise HTTPException(403, "local only")
        return push.propose()

    @prot.post("/execute")
    def ex(req: Exec, request: Request) -> dict:
        return push.execute(req.token, req.pin, client_ok=_local(request))

    @pub.get("/ui", response_class=HTMLResponse)
    def ui(request: Request) -> str:
        if not _local(request):
            raise HTTPException(403, "local only")
        return PAGE

    return [pub, prot]


_RX = re.compile(r"(?i)\b(push|publish|deploy|ship)\b.*\b(git|github|repo|repos|site|website|code|engine|changes|commits?)\b|\bgit push\b")


def handle(session: str, text: str) -> Optional[dict]:
    if not _RX.search(text or ""):
        return None
    st = push.status()
    if not st["enabled"]:
        return {"reply": "Pushing is not set up on this computer. " + st["how_to"], "blocks": [], "actions": []}
    p = push.propose()
    lines = []
    for pl in p.get("plans", []):
        lines.append(f'**{pl["name"]}**: ' + (f'{pl["ahead"]} commit(s) ready on {pl["branch"]}' + (" — BLOCKED by the secret scan" if pl.get("secrets") else "") if pl.get("ok") else pl.get("why", "")))
    reply = "Here is what would go out (nothing is pushed yet):\n" + "\n".join(lines) + ("\n\nOpen the confirmation page, check the commits, enter your PIN and press Push." if p.get("token") else "\n\n" + p.get("note", ""))
    acts = [{"type": "open", "label": "Review and push (PIN required)", "url": "/maintainer/ui"}] if p.get("token") else []
    return {"reply": reply, "blocks": [], "actions": acts}
