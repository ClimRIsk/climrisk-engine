/* ClimRisk resident agent dock — injected into every engine page. No external libraries, no external services. */
(function () {
  if (window.__criDock) return; window.__criDock = true;
  var FULL = !!window.__RESIDENT_FULL;
  var LS = window.localStorage;
  function lsGet(k, d) { try { var v = LS.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } }
  function lsSet(k, v) { try { LS.setItem(k, JSON.stringify(v)); } catch (e) {} }
  var session = lsGet('cri_session', null) || (function () { var s = 's' + Math.random().toString(36).slice(2, 10); lsSet('cri_session', s); return s; })();
  var msgs = lsGet('cri_dock_msgs', []);
  var open = FULL || lsGet('cri_dock_open', false);

  var css = '#cri-dock *{box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}' +
    '#cri-fab{position:fixed;right:18px;bottom:18px;z-index:99998;background:#0ea5e9;color:#04121f;border:0;border-radius:26px;padding:11px 18px;font-weight:700;font-size:14px;cursor:pointer;box-shadow:0 6px 24px rgba(0,0,0,.35)}' +
    '#cri-panel{position:fixed;right:18px;bottom:18px;width:420px;max-width:calc(100vw - 24px);height:620px;max-height:calc(100vh - 24px);z-index:99999;background:#0f172a;color:#e2e8f0;border:1px solid #1e293b;border-radius:14px;display:none;flex-direction:column;box-shadow:0 12px 48px rgba(0,0,0,.5)}' +
    '#cri-dock.full #cri-panel{position:static;width:100%;max-width:none;height:100vh;max-height:none;border-radius:0;display:flex}' +
    '#cri-hd{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid #1e293b;font-weight:700;font-size:14px}' +
    '#cri-hd span.s{font-weight:400;color:#94a3b8;font-size:11px;flex:1}' +
    '#cri-hd button{background:none;border:0;color:#94a3b8;cursor:pointer;font-size:12px}' +
    '#cri-log{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:10px}' +
    '.cm{max-width:92%;padding:9px 11px;border-radius:10px;font-size:13px;line-height:1.45;white-space:pre-wrap;word-wrap:break-word}' +
    '.cm.u{align-self:flex-end;background:#0ea5e9;color:#04121f}.cm.a{align-self:flex-start;background:#1e293b}' +
    '.cm table{border-collapse:collapse;margin-top:8px;font-size:11.5px;width:100%}.cm th,.cm td{border-bottom:1px solid #334155;padding:3px 5px;text-align:left}.cm th{color:#94a3b8}' +
    '.cm ul{margin:6px 0 0 16px;padding:0}.cm .note{margin-top:8px;color:#94a3b8;font-size:11.5px}' +
    '.act{display:inline-block;margin:6px 6px 0 0;background:#0b3b53;color:#7dd3fc;border:1px solid #155e75;border-radius:14px;padding:4px 10px;font-size:12px;cursor:pointer}' +
    '.act.pub{background:#14532d;color:#bbf7d0;border-color:#166534}' +
    '.post{margin-top:8px;border:1px solid #334155;border-radius:8px;padding:8px;background:#0b1220}.post textarea{width:100%;height:150px;background:#0b1220;color:#e2e8f0;border:1px solid #334155;border-radius:6px;font-size:12px;padding:6px}' +
    '.post img{width:100%;border-radius:6px;margin-top:6px}' +
    '#cri-chips{padding:0 12px 6px;display:flex;gap:6px;flex-wrap:wrap}#cri-chips .act{margin:0}' +
    '#cri-in{display:flex;gap:6px;padding:10px;border-top:1px solid #1e293b}#cri-in input{flex:1;background:#0b1220;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:9px;font-size:13px}' +
    '#cri-in button{background:#0ea5e9;border:0;border-radius:8px;color:#04121f;font-weight:700;padding:0 14px;cursor:pointer}' +
    '.spin{display:inline-block;width:10px;height:10px;border:2px solid #475569;border-top-color:#7dd3fc;border-radius:50%;animation:criS 1s linear infinite;margin-right:6px}@keyframes criS{to{transform:rotate(360deg)}}';
  var st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  var root = document.createElement('div'); root.id = 'cri-dock'; if (FULL) root.className = 'full';
  root.innerHTML = '<button id="cri-fab">Ask ClimRisk</button><div id="cri-panel"><div id="cri-hd">ClimRisk agent <span class="s">runs inside this engine</span>' +
    '<button id="cri-set">status</button>' + (FULL ? '' : '<button id="cri-x">✕</button>') + '</div><div id="cri-log"></div><div id="cri-chips"></div>' +
    '<div id="cri-in"><input id="cri-q" placeholder="Ask, e.g. build a 3D twin of 22.2645, 91.805" autocomplete="off"><button id="cri-go">Send</button></div></div>';
  document.body.appendChild(root);
  var $ = function (id) { return document.getElementById(id); };
  var panel = $('cri-panel'), log = $('cri-log'), fab = $('cri-fab');

  function key() { var h = (location.hash.match(/key=([^&]+)/) || [])[1]; if (h) lsSet('cri_api_key', decodeURIComponent(h)); return lsGet('cri_api_key', ''); }
  function api(path, method, body) {
    var k = key(); var h = { 'Content-Type': 'application/json' }; if (k) h['X-API-Key'] = k;
    return fetch(path, { method: method || 'GET', headers: h, body: body ? JSON.stringify(body) : undefined }).then(function (r) { return r.json().then(function (j) { return { ok: r.ok, status: r.status, j: j }; }); });
  }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function md(s) { return esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>'); }

  function save() { lsSet('cri_dock_msgs', msgs.slice(-60)); }
  function show(o) { open = o; panel.style.display = (o || FULL) ? 'flex' : 'none'; fab.style.display = (o || FULL) ? 'none' : 'block'; if (!FULL) lsSet('cri_dock_open', o); if (o) { render(); setTimeout(function () { $('cri-q').focus(); }, 50); } }
  fab.onclick = function () { show(true); }; if ($('cri-x')) $('cri-x').onclick = function () { show(false); };

  function blocksHtml(blocks) {
    return (blocks || []).map(function (b) {
      if (b.type === 'list') return '<ul>' + b.items.map(function (i) { return '<li>' + esc(i) + '</li>'; }).join('') + '</ul>';
      if (b.type === 'table') return '<table><tr>' + b.head.map(function (h) { return '<th>' + esc(h) + '</th>'; }).join('') + '</tr>' + b.rows.map(function (r) { return '<tr>' + r.map(function (c) { return '<td>' + esc(c) + '</td>'; }).join('') + '</tr>'; }).join('') + '</table>' + (b.caption ? '<div class="note">' + esc(b.caption) + '</div>' : '');
      if (b.type === 'note') return '<div class="note">' + esc(b.text) + '</div>';
      if (b.type === 'sources') return '<div class="note"><b>Sources</b><br>' + b.items.map(function (s) { return '[' + s.n + '] <a href="' + esc(s.url) + '" target="_blank" rel="noopener" style="color:#7cc4ff">' + esc(s.title) + '</a> · learned ' + esc(s.learned); }).join('<br>') + '</div>';
      if (b.type === 'files') return '<div style="margin-top:8px">' + b.items.map(function (f) { return '<a class="act" href="' + esc(f.url) + '" target="_blank" rel="noopener" style="display:inline-block;margin:0 6px 6px 0;text-decoration:none">' + esc(f.label) + '</a>'; }).join('') + '</div>';
      if (b.type === 'post') return '<div class="post"><textarea data-post="' + b.id + '">' + esc(b.text) + '</textarea>' + (b.has_image ? '<img src="' + b.image_url + (key() ? '' : '') + '" alt="post image">' : '') + '</div>';
      return '';
    }).join('');
  }
  function actionsHtml(m) {
    return (m.actions || []).map(function (a, i) {
      var cls = a.type === 'publish' ? 'act pub' : 'act';
      return '<span class="' + cls + '" data-i="' + i + '" data-mi="' + m._i + '">' + esc(a.label) + '</span>';
    }).join('');
  }
  function render() {
    log.innerHTML = msgs.map(function (m, i) {
      m._i = i;
      var inner = m.role === 'u' ? esc(m.text) : (m.pending ? '<span class="spin"></span>' : '') + md(m.text) + blocksHtml(m.blocks) + actionsHtml(m);
      return '<div class="cm ' + m.role + '">' + inner + '</div>';
    }).join('');
    log.scrollTop = log.scrollHeight;
    log.querySelectorAll('.act').forEach(function (el) { el.onclick = function () { act(msgs[+el.dataset.mi], +el.dataset.i); }; });
    log.querySelectorAll('.post img').forEach(function (img) {
      var k = key(); if (k && img.src.indexOf('blob:') < 0) { fetch(img.getAttribute('src'), { headers: { 'X-API-Key': k } }).then(function (r) { return r.blob(); }).then(function (b) { img.src = URL.createObjectURL(b); }); }
    });
  }
  function chips() {
    var c = ['What did you find while I was away?', 'Run the autopilot now', 'Scan this week\'s events', 'How do I protect my site?'];
    $('cri-chips').innerHTML = c.map(function (t) { return '<span class="act">' + esc(t) + '</span>'; }).join('');
    $('cri-chips').querySelectorAll('.act').forEach(function (el) { el.onclick = function () { send(el.textContent); }; });
  }

  function act(m, i) {
    var a = (m.actions || [])[i]; if (!a) return;
    if (a.type === 'open') { if (a.url) { window.location.href = a.url + (key() ? (a.url.indexOf('#') < 0 ? '#key=' + encodeURIComponent(key()) : '') : ''); } }
    else if (a.type === 'say') send(a.text);
    else if (a.type === 'intent') { send(a.text, a.intent); }
    else if (a.type === 'copy') { var ta = log.querySelector('textarea[data-post="' + a.post_id + '"]'); if (ta) { ta.select(); try { document.execCommand('copy'); } catch (e) {} push('a', 'Copied the post text.'); } }
    else if (a.type === 'publish') {
      var ta2 = log.querySelector('textarea[data-post="' + a.post_id + '"]'); var txt = ta2 ? ta2.value : '';
      if (!window.confirm('Publish this exact post (text and image) to LinkedIn now?\n\n' + txt.slice(0, 400))) return;
      var go = function () { return api('/resident/post/' + a.post_id + '/publish', 'POST', { approved: true }); };
      (txt ? api('/resident/post/' + a.post_id, 'PUT', { text: txt }) : Promise.resolve()).then(go).then(function (r) {
        push('a', r.j.ok ? 'Published to LinkedIn.' + (r.j.url ? ' ' + r.j.url : '') : 'Not published: ' + (r.j.error || (r.j.detail && r.j.detail.error) || r.status));
      });
    }
  }
  function push(role, text, extra) { var m = Object.assign({ role: role, text: text }, extra || {}); msgs.push(m); save(); render(); return m; }

  function poll(m, jid) {
    var t = setInterval(function () {
      api('/resident/job/' + jid).then(function (r) {
        var j = r.j; if (j.status === 'running') return;
        clearInterval(t); m.pending = false;
        if (j.status === 'done') { m.text = j.reply || 'Done.'; m.blocks = j.blocks; m.actions = (m.actions || []).concat(j.actions || []); }
        else { m.text = 'The job failed: ' + (j.error || j.status); }
        save(); render();
      });
    }, 2500);
  }
  function send(text, intent) {
    text = (text || '').trim(); if (!text) return;
    push('u', text);
    var m = push('a', 'Thinking…', { pending: true });
    api('/resident/chat', 'POST', { text: text, session: session, intent: intent || null }).then(function (r) {
      if (!r.ok) { m.pending = false; m.text = r.status === 403 ? 'The engine needs an API key. Add #key=YOUR_KEY to the page address once.' : 'Error ' + r.status; save(); render(); return; }
      var j = r.j; m.text = j.reply; m.blocks = j.blocks; m.actions = j.actions; m.pending = !!j.job; save(); render();
      if (j.job) poll(m, j.job);
    }).catch(function (e) { m.pending = false; m.text = 'Could not reach the engine: ' + e; save(); render(); });
  }
  $('cri-go').onclick = function () { var v = $('cri-q').value; $('cri-q').value = ''; send(v); };
  $('cri-q').onkeydown = function (e) { if (e.key === 'Enter') $('cri-go').onclick(); };
  $('cri-set').onclick = function () {
    api('/resident/status').then(function (r) {
      var s = r.j; if (!s.model) { push('a', 'Status unavailable.'); return; }
      push('a', 'Language model: ' + s.model.n_classes + ' intents, held-out phrasing accuracy ' + Math.round(100 * (s.model.heldout_model_only || 0)) + '% (model alone), grouped CV ' + Math.round(100 * (s.model.grouped_cv_mean || 0)) + '%. Feedback rows waiting: ' + s.feedback_rows + '. LinkedIn publishing: ' + (s.linkedin.configured ? 'configured' : 'not configured (drafts only)') + '. Facts remembered: ' + s.facts + '.',
        { actions: [{ type: 'retrain', label: 'Retrain from my corrections' }] });
      var last = msgs[msgs.length - 1]; last.actions[0].type = 'say'; last.actions[0].text = 'retrain';
    });
  };
  var _send = send; send = function (t, i) { if (t.toLowerCase() === 'retrain') { api('/resident/retrain', 'POST').then(function (r) { var j = r.j; push('a', 'Retrained on ' + j.n_train + ' examples (' + j.n_feedback + ' from your corrections). Held-out phrasing accuracy (never-seen templates): ' + Math.round(100 * j.heldout_templates) + '%; grouped CV ' + Math.round(100 * j.grouped_cv.mean) + '%.'); }); return; } _send(t, i); };

  if (!msgs.length) msgs.push({ role: 'a', text: 'I run inside this engine — no outside software. I can build 3D twins, find and verify protection, build scenarios, write story reports, research live hazards and draft LinkedIn posts (published only when you approve). What do you want to look at?' });
  // ── autopilot: the agent works in the background; show what it found ──
  var apState = null;
  function ago(ts) { if (!ts) return 'not yet'; var m = Math.round((Date.now() / 1000 - ts) / 60); return m < 1 ? 'just now' : m < 90 ? m + ' min ago' : Math.round(m / 60) + ' h ago'; }
  function pullInbox(announce) {
    api('/resident/inbox?unread=true').then(function (r) {
      if (!r.ok || !r.j.items) return; apState = r.j.status;
      var n = r.j.unread; fab.textContent = n ? 'Ask ClimRisk · ' + n + ' new' : 'Ask ClimRisk';
      var hd = $('cri-hd').querySelector('span.s'); if (hd && apState) hd.textContent = apState.enabled ? 'autopilot on · last cycle ' + ago(apState.last_run) + (apState.running ? ' · working…' : '') : 'autopilot off';
      if (n && (open || FULL) && announce !== false) {
        r.j.items.slice().reverse().forEach(function (it) {
          msgs.push({ role: 'a', text: '🛰 ' + it.title + '\n' + it.detail, blocks: it.blocks, actions: it.actions });
        });
        save(); render(); api('/resident/inbox/read', 'POST', { ids: r.j.items.map(function (i) { return i.id; }) }).then(function () { fab.textContent = 'Ask ClimRisk'; });
      }
    });
  }
  var _show = show; show = function (o) { _show(o); if (o) pullInbox(true); };
  pullInbox(true); setInterval(function () { pullInbox(true); }, 60000);
  chips(); show(open);
})();
