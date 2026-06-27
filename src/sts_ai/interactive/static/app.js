"use strict";
// Interactive Rollout Studio — vanilla JS client (no bundler, offline).

const S = {
  config: null,
  sid: null,        // current session id
  viewing: 0,       // decision index currently displayed
  nDecisions: 0,
  framings: {},
  prompts: {},
  es: null,         // active EventSource (streaming)
};

const $ = (id) => document.getElementById(id);

async function api(path, { method = "GET", body = null } = {}) {
  const opts = { method, headers: {} };
  if (body != null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail;
    try { detail = (await r.json()).detail; } catch (e) { detail = r.statusText; }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return r.json();
}

function setStatus(msg, cls = "") {
  const el = $("status-line");
  el.textContent = msg || "";
  el.className = "status-line " + cls;
}
async function guard(fn, busyMsg) {
  try {
    if (busyMsg) setStatus(busyMsg, "busy");
    const out = await fn();
    if (busyMsg) setStatus("");
    return out;
  } catch (e) {
    setStatus(e.message, "err");
    console.error(e);
  }
}

// ---------- init ----------
async function init() {
  S.config = await guard(() => api("/api/config"), "loading…");
  if (S.config) {
    $("board-css").textContent = S.config.board_css || "";
    $("cfg-model").textContent =
      `backend: ${S.config.default_model_backend} · model: ${S.config.default_model_id || "(agent default)"}` +
      (S.config.mlx_available ? "" : "  ⚠ mlx not installed");
  }
  await refreshTemplates();
  await refreshSessions();
  wireEvents();
}

async function refreshTemplates() {
  S.framings = (await guard(() => api("/api/templates/framings"))) || {};
  S.prompts = (await guard(() => api("/api/templates/prompts"))) || {};
  fillSelect($("framing-presets"), Object.keys(S.framings));
  fillSelect($("prompt-presets"), Object.keys(S.prompts));
}
function fillSelect(sel, names) {
  sel.innerHTML = "";
  for (const n of names) {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    sel.appendChild(o);
  }
}

// ---------- sessions tree ----------
async function refreshSessions() {
  const sessions = (await guard(() => api("/api/sessions"))) || [];
  renderTree(sessions);
}
function renderTree(sessions) {
  const byId = {};
  sessions.forEach((s) => (byId[s.session_id] = { ...s, children: [] }));
  const roots = [];
  sessions.forEach((s) => {
    const node = byId[s.session_id];
    if (s.parent_id && byId[s.parent_id]) byId[s.parent_id].children.push(node);
    else roots.push(node);
  });
  const host = $("session-tree");
  host.innerHTML = "";
  if (!roots.length) { host.innerHTML = '<em class="dim">none yet</em>'; return; }
  host.appendChild(renderNodes(roots));
}
function renderNodes(nodes) {
  const ul = document.createElement("ul");
  for (const n of nodes) {
    const li = document.createElement("li");
    const div = document.createElement("div");
    div.className = "node" + (n.session_id === S.sid ? " active" : "");
    const bp = n.branch_point != null ? `@${n.branch_point}` : "";
    div.innerHTML =
      `${n.label || n.session_id} <span class="meta">seed ${n.world_seed} · ${n.combat_control} · ` +
      `${n.n_decisions}d ${bp} · ${n.session_status || ""}</span>`;
    div.onclick = () => selectSession(n.session_id);
    li.appendChild(div);
    if (n.children.length) li.appendChild(renderNodes(n.children));
    ul.appendChild(li);
  }
  return ul;
}

// ---------- session view ----------
async function selectSession(id) {
  closeStream();
  const view = await guard(() => api(`/api/sessions/${id}`), "loading session…");
  if (!view) return;
  S.sid = id;
  loadEditorFromView(view);
  applyView(view, view.decision_index);
  await refreshSessions();
}

function loadEditorFromView(view) {
  // Populate editor + config when switching sessions (don't clobber on every render).
  if (view.framing != null) $("framing-text").value = view.framing;
  if (view.prompt_template != null) $("prompt-text").value = view.prompt_template;
  $("use-advanced").checked = !!view.use_advanced_template;
  if (view.temperature != null) $("cfg-temp").value = view.temperature;
  if (view.max_tokens != null) $("cfg-maxtok").value = view.max_tokens;
  $("cfg-thinking").checked = !!view.thinking;
}

function applyView(view, atIndex) {
  S.nDecisions = view.n_decisions != null ? view.n_decisions : S.nDecisions;
  S.viewing = atIndex != null ? atIndex : (view.decision_index != null ? view.decision_index : S.viewing);

  // board
  const board = $("board");
  if (view.board_html) board.innerHTML = view.board_html;
  else if (view.done) board.innerHTML = `<em class="dim">Run finished (${view.stopped_reason || view.status}).</em>`;
  else board.innerHTML = `<em class="dim">${view.status || ""}</em>`;

  // session info
  const isFrontier = S.viewing >= S.nDecisions;
  $("session-info").innerHTML =
    `<b>${view.label || S.sid}</b><br>seed ${view.world_seed} · ${view.combat_control}` +
    `<br>${S.nDecisions} decisions · ${view.session_status || ""}` +
    (view.stopped_reason ? `<br><span class="dim">${view.stopped_reason}</span>` : "") +
    `<br><span class="dim">${isFrontier ? "at frontier (live)" : "viewing decision " + S.viewing + " (read-only)"}</span>`;

  // scrubber
  const scrub = $("scrub");
  scrub.max = S.nDecisions;
  scrub.value = Math.min(S.viewing, S.nDecisions);
  $("scrub-label").textContent = S.viewing;

  // legal actions (clickable only at the live frontier)
  renderLegalActions(view, isFrontier);

  // disable act controls when terminal
  const terminal = !!view.done;
  ["btn-step", "btn-stream", "btn-sample"].forEach((b) => ($(b).disabled = terminal || !isFrontier));
}

function renderLegalActions(view, isFrontier) {
  const host = $("legal-actions");
  host.innerHTML = "";
  const actions = view.legal_actions || [];
  if (!actions.length) { host.innerHTML = '<em class="dim">no legal actions</em>'; return; }
  if (!isFrontier) {
    host.innerHTML = '<em class="dim">viewing a past decision — go to frontier to act</em>';
    return;
  }
  actions.forEach((a) => {
    const b = document.createElement("button");
    b.className = "la";
    b.textContent = `${a.index}: ${a.description}`;
    b.title = "commit this action (as user choice)";
    b.onclick = () => stepUser(a.index);
    host.appendChild(b);
  });
}

// ---------- stepping ----------
async function stepUser(actionIndex) {
  const out = await guard(
    () => api(`/api/sessions/${S.sid}/step`, { method: "POST", body: { method: "user", action_index: actionIndex } }),
    "stepping…"
  );
  if (out) afterStep(out);
}
async function step() {
  const method = $("method").value;
  const n = parseInt($("n-steps").value || "1", 10);
  if (method === "user") { setStatus("pick a legal action above for user method", "err"); return; }
  const out = await guard(
    () => api(`/api/sessions/${S.sid}/step`, { method: "POST", body: { method, n } }),
    "stepping…"
  );
  if (out) afterStep(out);
}
function afterStep(out) {
  $("candidates").innerHTML = "";
  if (out.status && out.status !== "ok" && out.status !== "terminal") setStatus(out.status, "err");
  applyView(out.view, out.view ? out.view.decision_index : null);
  refreshSessions();
}

// ---------- streaming ----------
function closeStream() {
  if (S.es) { S.es.close(); S.es = null; }
  $("stream-wrap").classList.add("hidden");
}
function streamModel() {
  if (!S.sid) return;
  const method = $("method").value === "user" ? "model" : $("method").value;
  closeStream();
  $("stream").textContent = "";
  $("stream-wrap").classList.remove("hidden");
  setStatus("generating…", "busy");
  const es = new EventSource(`/api/sessions/${S.sid}/stream_step?method=${encodeURIComponent(method)}`);
  S.es = es;
  es.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.type === "token") {
      $("stream").textContent += msg.text;
      $("stream").scrollTop = $("stream").scrollHeight;
    } else if (msg.type === "done") {
      es.close(); S.es = null;            // close BEFORE the server drops the conn (no auto-reconnect)
      setStatus(msg.status === "ok" || msg.status === "terminal" ? "" : (msg.status || ""), msg.status === "ok" ? "" : "err");
      if (msg.view) applyView(msg.view, msg.view.decision_index);
      refreshSessions();
    }
  };
  es.onerror = () => { closeStream(); setStatus("stream error", "err"); };
}

// ---------- sampling ----------
async function sample() {
  const method = $("method").value;
  const k = parseInt($("k-samples").value || "1", 10);
  const out = await guard(
    () => api(`/api/sessions/${S.sid}/sample`, { method: "POST", body: { method, k } }),
    "sampling…"
  );
  if (out) renderCandidates(out.candidates || []);
}
function renderCandidates(cands) {
  const host = $("candidates");
  host.innerHTML = "";
  cands.forEach((c) => {
    const div = document.createElement("div");
    div.className = "cand";
    const valid = c.valid === false ? " ⚠ invalid" : "";
    div.innerHTML =
      `<div class="hd"><span><b>${c.action_index}</b>: ${escapeHtml(c.description || "")}${valid}</span></div>` +
      (c.reasoning ? `<div class="why">${escapeHtml(c.reasoning)}</div>` : "") +
      (c.thinking ? `<details class="think"><summary>thinking</summary>${escapeHtml(c.thinking)}</details>` : "");
    const btn = document.createElement("button");
    btn.textContent = "use this action";
    btn.onclick = () => stepUser(c.action_index);
    div.querySelector(".hd").appendChild(btn);
    host.appendChild(div);
  });
  if (!cands.length) host.innerHTML = '<em class="dim">no candidates</em>';
}

// ---------- branching / navigation ----------
async function branchHere() {
  if (!S.sid) return;
  const child = await guard(
    () => api(`/api/sessions/${S.sid}/branch`, { method: "POST", body: { at: S.viewing } }),
    "branching…"
  );
  if (child) { await refreshSessions(); selectSession(child.session_id); }
}
async function navigate(k) {
  if (!S.sid) return;
  k = Math.max(0, Math.min(k, S.nDecisions));
  const view = await guard(() => api(`/api/sessions/${S.sid}/view?at=${k}`));
  if (view) applyView(view, k);
}

// ---------- prompt / config editing ----------
async function applyFraming() {
  const body = { framing: $("framing-text").value, use_advanced: $("use-advanced").checked };
  if ($("use-advanced").checked) body.template = $("prompt-text").value;
  const view = await guard(() => api(`/api/sessions/${S.sid}/framing`, { method: "PUT", body }), "applying…");
  if (view) applyView(view, view.decision_index);
}
async function saveFraming() {
  const name = $("framing-save-name").value.trim();
  if (!name) { setStatus("enter a name to save", "err"); return; }
  await guard(() => api("/api/templates/framings", { method: "POST", body: { name, text: $("framing-text").value } }), "saving…");
  await refreshTemplates();
}
async function savePrompt() {
  const name = $("prompt-save-name").value.trim();
  if (!name) { setStatus("enter a name to save", "err"); return; }
  await guard(() => api("/api/templates/prompts", { method: "POST", body: { name, text: $("prompt-text").value } }), "saving…");
  await refreshTemplates();
}
async function preview() {
  const body = { framing: $("framing-text").value, use_advanced: $("use-advanced").checked, template: $("prompt-text").value };
  const out = await guard(() => api(`/api/sessions/${S.sid}/preview_prompt`, { method: "POST", body }), "rendering…");
  if (out) { $("prompt-preview").textContent = out.prompt || "(none)"; $("prompt-preview").classList.remove("hidden"); }
}
async function pushModelConfig() {
  if (!S.sid) return;
  await guard(() => api(`/api/sessions/${S.sid}/config`, {
    method: "PUT",
    body: {
      temperature: parseFloat($("cfg-temp").value),
      max_tokens: parseInt($("cfg-maxtok").value, 10),
      thinking: $("cfg-thinking").checked,
    },
  }));
}

// ---------- new session ----------
async function createSession() {
  const body = {
    world_seed: parseInt($("ns-seed").value, 10),
    ascension: parseInt($("ns-asc").value, 10),
    combat_control: $("ns-combat").value,
    max_act: parseInt($("ns-maxact").value, 10),
    battle_simulations: parseInt($("ns-sims").value, 10),
    label: $("ns-label").value,
    model_backend: S.config ? S.config.default_model_backend : "mlx",
    model_id: S.config ? S.config.default_model_id : null,
  };
  const view = await guard(() => api("/api/sessions", { method: "POST", body }), "creating (replaying)…");
  if (view) { S.sid = view.session_id; loadEditorFromView(view); applyView(view, view.decision_index); await refreshSessions(); }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- wiring ----------
function wireEvents() {
  $("btn-new").onclick = () => $("new-dialog").showModal();
  $("new-form").addEventListener("close", () => {});
  $("new-dialog").addEventListener("close", (e) => {
    if ($("new-dialog").returnValue === "create") createSession();
  });

  $("btn-step").onclick = step;
  $("btn-stream").onclick = streamModel;
  $("btn-sample").onclick = sample;
  $("btn-cancel-stream").onclick = () => { closeStream(); setStatus("cancelled"); };
  $("btn-branch").onclick = branchHere;

  $("btn-prev").onclick = () => navigate(S.viewing - 1);
  $("btn-next").onclick = () => navigate(S.viewing + 1);
  $("btn-frontier").onclick = () => navigate(S.nDecisions);
  $("scrub").addEventListener("input", (e) => navigate(parseInt(e.target.value, 10)));

  $("btn-apply-framing").onclick = applyFraming;
  $("btn-apply-prompt").onclick = applyFraming; // advanced apply uses same endpoint
  $("btn-save-framing").onclick = saveFraming;
  $("btn-save-prompt").onclick = savePrompt;
  $("btn-load-framing").onclick = () => { $("framing-text").value = S.framings[$("framing-presets").value] || ""; };
  $("btn-load-prompt").onclick = () => { $("prompt-text").value = S.prompts[$("prompt-presets").value] || ""; };
  $("btn-preview").onclick = preview;

  ["cfg-temp", "cfg-maxtok", "cfg-thinking"].forEach((id) => $(id).addEventListener("change", pushModelConfig));

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== tab.dataset.tab));
    };
  });
}

init();
