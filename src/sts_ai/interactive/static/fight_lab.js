const $ = id => document.getElementById(id);
let collection, rows = [], page = 0, current = null, trajectories = [];
let count = 0, at = 0, timer = null, request = 0, runTarget = null;
let latestJob = null, submitting = false;

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}
const post = (url, body = {}) => api(url, {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
});
function guard(fn) {
  return async (...args) => {
    try { $('error').textContent = ''; await fn(...args); }
    catch (error) { $('error').textContent = error.message; stop(); }
  };
}
function params() {
  const p = new URLSearchParams();
  for (const [key, value] of new FormData($('filters'))) {
    if (value || key === 'phase') p.set(key, key === 'battle_start' ? 'true' : value);
  }
  return p;
}
function stop() { clearInterval(timer); timer = null; $('play').textContent = 'Play'; }
function summary() {
  $('summary').textContent = `${collection.sources.length} verified source runs · ${collection.n_states} states · ${trajectories.length} policy trajectories`;
}
function runControls() {
  const busy = submitting || latestJob?.status === 'running';
  const canRun = runTarget?.phase === 'combat';
  $('runModel').disabled = busy || !canRun;
  $('runTarget').textContent = canRun
    ? `${runTarget.enemies.join(' + ')} · World ${runTarget.world_seed} · Floor ${runTarget.floor} · Turn ${runTarget.turn} · Decision ${runTarget.index}`
    : 'Select a combat position in the source rollout to run Gemma.';
  $('exportState').setAttribute('aria-disabled', String(!canRun));
  if (canRun) $('exportState').href = '/api/states?' + new URLSearchParams({state_id: runTarget.state_id, download: 'true'});
  else $('exportState').removeAttribute('href');
  $('cancelRun').hidden = latestJob?.status !== 'running';
  $('showRun').hidden = !latestJob?.trajectory || latestJob.status === 'running';
  if (latestJob) {
    const result = latestJob.result;
    $('runStatus').textContent = latestJob.status === 'running'
      ? `Running ${latestJob.state_id} · ${latestJob.decisions} decisions saved${latestJob.decisions ? '' : ' · preparing the fight and loading Gemma…'}`
      : `${latestJob.state_id} · ${latestJob.status}${result?.metrics ? ` · ${result.metrics.label} · net HP loss ${result.metrics.hp_loss}` : ''}${result?.stopped_reason ? ` · ${result.stopped_reason}` : ''}`;
    $('runError').hidden = !latestJob.error;
    $('runError').textContent = latestJob.error || '';
  }
}
function renderMatches() {
  $('matches').replaceChildren();
  for (const row of rows.slice(page * 40, page * 40 + 40)) {
    const button = document.createElement('button');
    button.className = 'match' + (current?.state_id === row.state_id ? ' active' : '');
    button.textContent = `${row.enemies.join(' + ') || row.phase} · ${row.hp}/${row.max_hp} HP`;
    const small = document.createElement('small');
    small.textContent = `World ${row.world_seed} · Floor ${row.floor} · ${row.turn ? 'Turn ' + row.turn : 'OOC'} · ${row.split} · d${row.index}`;
    button.append(document.createElement('br'), small);
    button.onclick = guard(() => choose(row));
    $('matches').append(button);
  }
  $('status').textContent = `${rows.length} matching states · page ${page + 1}/${Math.max(1, Math.ceil(rows.length / 40))}`;
  $('pagePrev').disabled = page === 0;
  $('pageNext').disabled = (page + 1) * 40 >= rows.length;
}
async function filter() {
  stop();
  rows = (await api('/api/states?' + params())).states;
  page = 0;
  $('export').href = '/api/states?' + params() + '&download=true';
  renderMatches();
  if (rows.length) await choose(rows[0]);
  else {
    ++request; current = null; runTarget = null; count = 0;
    $('trace').replaceChildren(); $('board').textContent = 'No matching states.';
    $('raw').textContent = ''; $('traceInfo').textContent = ''; $('position').textContent = '';
    runControls();
  }
}
function traceOptions(selected) {
  $('trace').replaceChildren();
  const add = (value, text) => {
    const option = document.createElement('option'); option.value = value; option.textContent = text;
    $('trace').append(option);
  };
  add(current.source, 'Source heuristic rollout');
  for (const trajectory of trajectories.filter(t => t.state_id === current.state_id)) {
    add(trajectory.path, `${trajectory.label} · ${trajectory.metrics.label} · HP loss ${trajectory.metrics.hp_loss}`);
  }
  if (selected && [...$('trace').options].some(o => o.value === selected)) $('trace').value = selected;
}
async function choose(row) {
  stop(); current = row; runTarget = null; runControls(); renderMatches(); traceOptions();
  await view(row.index);
}
async function view(index) {
  if (!$('trace').value) return;
  const seq = ++request, source = $('trace').value;
  runTarget = null; runControls();
  const v = await api('/api/view?' + new URLSearchParams({source, index}));
  if (seq !== request) return;
  at = v.index; count = v.count; runTarget = v.source_state;
  // Scrubbing the source changes the actual position, including which model runs belong to it.
  if (runTarget && current.state_id !== runTarget.state_id) {
    current = runTarget; renderMatches(); traceOptions(source);
  }
  $('timeline').max = count - 1; $('timeline').value = at;
  $('position').textContent = `${at + 1} / ${count}`;
  $('board').innerHTML = v.board_html; $('raw').textContent = JSON.stringify(v.record, null, 2);
  $('traceInfo').textContent = `${$('trace').selectedOptions[0].textContent} · stopped: ${v.outcome}`;
  $('prev').disabled = at === 0; $('next').disabled = at === count - 1;
  if (at === count - 1) stop();
  runControls();
}
async function showJob(job) {
  const selected = await api('/api/states?' + new URLSearchParams({state_id: job.state_id}));
  trajectories = await api('/api/trajectories'); summary();
  await choose(selected.states[0]);
  if (job.trajectory && trajectories.some(t => t.path === job.trajectory)) {
    $('trace').value = job.trajectory; await view(0);
  }
}
async function pollJobs() {
  const previous = latestJob;
  const jobs = await api('/api/runs');
  latestJob = jobs.find(j => j.status === 'running') || jobs[0] || null;
  runControls();
  if (previous?.status === 'running' && latestJob?.job_id === previous.job_id && latestJob.status !== 'running') {
    trajectories = await api('/api/trajectories'); summary();
    if (latestJob.trajectory && current?.state_id === latestJob.state_id) await showJob(latestJob);
    else if (current) traceOptions($('trace').value);
  }
}
$('runModel').onclick = guard(async () => {
  if (submitting || latestJob?.status === 'running' || runTarget?.phase !== 'combat') return;
  const stateId = runTarget.state_id;
  stop(); submitting = true; runControls();
  try { latestJob = await post('/api/runs', {state_id: stateId}); }
  finally { submitting = false; runControls(); }
});
$('cancelRun').onclick = guard(async () => {
  if (latestJob) latestJob = await post(`/api/runs/${latestJob.job_id}/cancel`);
  runControls();
});
$('showRun').onclick = guard(() => showJob(latestJob));
$('filters').onsubmit = guard(async event => { event.preventDefault(); await filter(); });
$('pagePrev').onclick = () => { page--; renderMatches(); };
$('pageNext').onclick = () => { page++; renderMatches(); };
$('trace').onchange = guard(() => { stop(); return view($('trace').value === current.source ? current.index : 0); });
$('prev').onclick = guard(() => { stop(); return view(Math.max(0, at - 1)); });
$('next').onclick = guard(() => { stop(); return view(Math.min(count - 1, at + 1)); });
$('timeline').oninput = guard(() => { stop(); return view(Number($('timeline').value)); });
$('play').onclick = () => {
  if (timer) return stop();
  if (!count) return;
  $('play').textContent = 'Pause'; timer = setInterval(guard(() => view(Math.min(count - 1, at + 1))), 1000);
};
guard(async () => {
  collection = await api('/api/collection'); trajectories = await api('/api/trajectories');
  const style = document.createElement('style'); style.textContent = collection.board_css; document.head.append(style);
  for (const enemy of collection.enemies) { const option = document.createElement('option'); option.value = enemy; $('enemies').append(option); }
  summary(); $('reports').textContent = JSON.stringify(collection.reports, null, 2);
  await filter(); await pollJobs();
  // Poll status only; refresh/reconnect never submits another run.
  async function poll() { await guard(pollJobs)(); setTimeout(poll, 1500); }
  setTimeout(poll, 1500);
})();
