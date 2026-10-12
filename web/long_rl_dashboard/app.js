"use strict";

// All numeric helpers keep absent measurements absent. No dependency or CDN.
const valid = value => typeof value === "number" && Number.isFinite(value);
const avg = values => {
  const available = values.filter(valid);
  return available.length ? available.reduce((sum, v) => sum + v, 0) / available.length : null;
};
function percentile(values, fraction) {
  const sorted = values.filter(valid).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * fraction;
  const low = Math.floor(position), high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
function rolling(points, windowSize) {
  return points.map((point, index) => {
    const window = points.slice(Math.max(0, index - windowSize + 1), index + 1);
    return {...point, y: valid(point.y) ? avg(window.map(p => p.y)) : null,
      contributors: window.filter(p => valid(p.y)).length};
  });
}
function histogram(values, count) {
  const available = values.filter(valid);
  if (!available.length) return {bins: [], count: 0, missing: values.length};
  let low = Math.min(...available), high = Math.max(...available);
  if (low === high) { low -= .5; high += .5; }
  const width = (high - low) / count;
  const bins = Array.from({length: count}, (_, index) => ({low: low + index * width,
    high: low + (index + 1) * width, count: 0}));
  for (const value of available) bins[Math.min(count - 1, Math.floor((value - low) / width))].count++;
  return {bins, count: available.length, missing: values.length - available.length};
}
function numericEligible(row, includeCuts = false) {
  if (row.requested_occurrence === false || row.coverage_valid === false || row.fault || ["unstarted", "missing"].includes(row.outcome) || row.unstarted) return false;
  if (!includeCuts && (row.truncated || row.censored)) return false;
  return row.numeric_eligible === true || (includeCuts && (row.truncated || row.censored) && row.coverage_valid === true);
}
function outcomeClass(row) {
  if (row.fault) return "fault";
  if (row.outcome === "unstarted" || row.unstarted) return "unstarted";
  if (row.outcome === "missing") return "other";
  if (row.truncated || row.censored) return "cutoff";
  if (["victory", "act_clear", "battle_victory"].includes(row.outcome)) return "success";
  if (row.outcome === "defeat") return "defeat";
  return "other";
}
function perRoundEpisodes(episodes, includeCuts) {
  const groups = new Map();
  for (const episode of episodes) {
    if (episode.requested_occurrence === false) continue;
    if (!valid(episode.round)) continue;
    if (!groups.has(episode.round)) groups.set(episode.round, []);
    groups.get(episode.round).push(episode);
  }
  return [...groups].sort((a,b) => a[0] - b[0]).map(([round, requested]) => {
    const numeric = requested.filter(row => numericEligible(row, includeCuts));
    const success = requested.filter(row => outcomeClass(row) === "success").length;
    return {round, n: requested.length, numeric_n: numeric.length,
      episode_return: avg(numeric.map(row => row.reward)),
      episode_floor: avg(numeric.map(row => row.floor)),
      episode_floor_progress: avg(numeric.map(row => row.floor_progress)),
      episode_hp_cost: avg(numeric.map(row => row.hp_cost)),
      episode_end_hp: avg(numeric.map(row => row.end_hp)),
      episode_end_hp_fraction: avg(numeric.map(row => row.end_hp_fraction)),
      episode_max_act: avg(numeric.map(row => row.max_act_reached)),
      episode_steps: avg(numeric.map(row => row.steps)),
      episode_success: success / requested.length,
      episode_fault_rate: requested.filter(row => outcomeClass(row) === "fault").length / requested.length,
      episode_cut_rate: requested.filter(row => outcomeClass(row) === "cutoff").length / requested.length};
  });
}
function applySavedOnce(saved, apply) {
  if (saved && typeof saved === "object") apply(saved);
  return null;
}
const dashboardMath = {applySavedOnce, valid, avg, percentile, rolling, histogram, numericEligible, outcomeClass, perRoundEpisodes};
if (typeof document === "undefined") globalThis.dashboardMath = dashboardMath;

if (typeof document !== "undefined") {
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const palette = ["#2563eb", "#e38919", "#179180", "#8656c3", "#de6773", "#73879e", "#48a5c8"];
const outcomeColors = {success: "#25937b", defeat: "#9baec6", fault: "#df685b", cutoff: "#e7ab3e", unstarted: "#8359bb", other: "#d4ddea"};
const state = {data: null, loading: false, range: null, rangeBounds: null, episodes: [], episodeResponse: null,
  episodeLoading: false, episodeError: null, episodeKey: null, episodeAbort: null, episodeToken: 0,
  drag: null, scatterDomain: null, scatterPoints: [], interactionTimer: null};
const fmt = (v, digits = 2) => valid(v) ? v.toLocaleString(undefined, {maximumFractionDigits: digits}) : "Unavailable";
const pct = v => valid(v) ? `${fmt(v * 100, 1)}%` : "Unavailable";
const hours = v => valid(v) ? `${fmt(v / 3600, 2)} h` : "Unavailable";
const age = v => valid(v) ? v < 120 ? `${Math.round(v)} s` : `${fmt(v / 60, 1)} min` : "Unavailable";
const utc = v => valid(v) ? new Date(v * 1000).toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC") : "Not recorded";
const shortSHA = sha => typeof sha === "string" ? sha.slice(0, 12) : "Unavailable";
const time = v => valid(v) ? new Date(v * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"}) : "—";
const metricDefs = {
  episode_return: {label: "Mean return · filtered episodes", episode: true},
  episode_floor: {label: "Mean absolute ending floor · filtered", episode: true},
  episode_floor_progress: {label: "Mean floors advanced · filtered", episode: true},
  episode_end_hp: {label: "Mean endpoint HP · filtered", episode: true},
  episode_end_hp_fraction: {label: "Mean endpoint HP fraction · filtered", episode: true, percent: true},
  episode_max_act: {label: "Mean max observed act · filtered", episode: true},
  episode_hp_cost: {label: "Mean net HP cost · filtered", episode: true},
  episode_steps: {label: "Mean episode steps · filtered", episode: true},
  episode_success: {label: "Target success / requested · filtered", episode: true, percent: true},
  episode_fault_rate: {label: "Faults / requested · filtered", episode: true, percent: true},
  episode_cut_rate: {label: "Cutoffs / requested · filtered", episode: true, percent: true},
  policy_loss: {label: "Policy loss · global PPO"}, value_loss: {label: "Value loss · global PPO"},
  approx_kl: {label: "Approximate KL · global PPO"}, entropy: {label: "Entropy · global PPO"},
  clip_fraction: {label: "Clip fraction · global PPO", percent: true},
  gradient_norm: {label: "Gradient norm · if logged"}, ratio_mean: {label: "Ratio mean · global PPO"},
  rejected_kl: {label: "Rejected KL · if logged"},
  transitions_per_second: {label: "Valid transitions / job second"},
  rollout_transitions_per_second: {label: "Valid transitions / rollout second"},
  steps: {label: "Accepted optimizer steps / job"}, transitions: {label: "Valid transitions / job"},
  rollout_seconds: {label: "Rollout seconds / job"}, update_seconds: {label: "Update seconds / job"},
  rejected_minibatches: {label: "Rejected minibatches / job"}, early_stop_rounds: {label: "Early-stop update rounds / job"}
};
const scatterDefs = {reward: "Undiscounted return", floor: "Absolute ending floor", floor_progress: "Floors advanced (end − start)", start_floor: "Starting floor", horizon: "Task horizon", start_hp: "Task reset HP", end_hp: "Endpoint HP", end_hp_fraction: "Endpoint HP fraction", max_act_reached: "Maximum observed act", hp_cost: "Net HP cost", steps: "Episode decisions / steps", end_act: "Ending act", start_act: "Starting act", goal_act: "Goal act"};
const savedIDs = ["performancePanel", "episodePanel", "healthMetric", "leftMetric", "rightMetric", "rolling", "rangePreset", "follow", "rawLines", "episodeKind", "source", "startAct", "goalAct", "component", "session", "outcome", "startFloorMin", "startFloorMax", "horizon", "snapshotRole", "includeCuts", "bins", "floorHistMetric", "scatterX", "scatterY", "scatterGroup", "auto"];
let selectedCampaign = "v2";
try {const savedCampaign=localStorage.getItem("sts-rl-condition");if(["v1","v2"].includes(savedCampaign))selectedCampaign=savedCampaign;}catch(_){}
$("campaign").value=selectedCampaign;
const storageKey=()=>`sts-rl-lab-v2:${selectedCampaign}`;
let restored = null;
try { restored = JSON.parse(localStorage.getItem(storageKey()) || "null"); } catch (_) { /* Storage may be unavailable. */ }
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function cards(id, values) {
  $(id).replaceChildren(...values.map(([label, value, note]) => {
    const node = el("div", undefined, "card");
    node.append(el("span", label), el("strong", value), el("small", note));
    return node;
  }));
}
function details(id, values) {
  $(id).replaceChildren(...values.map(([label, value]) => {
    const node = el("div", label); node.append(el("strong", value)); return node;
  }));
}
function table(id, headings, rows) {
  const target = $(id), head = el("thead"), headRow = el("tr"), body = el("tbody");
  headings.forEach(text => headRow.append(el("th", text))); head.append(headRow);
  rows.forEach(values => {
    const row = el("tr");
    values.forEach((value, index) => {
      const cell = el("td");
      if (value instanceof Node) cell.append(value); else cell.textContent = value ?? "—";
      if (index === 0) cell.className = "name";
      row.append(cell);
    });
    body.append(row);
  });
  target.replaceChildren(head, body);
}
function saveChoices() {
  const choices = Object.fromEntries(savedIDs.map(id => [id, $(id).type === "checkbox" ? $(id).checked : $(id).value]));
  choices.range = state.range;
  try { localStorage.setItem(storageKey(), JSON.stringify(choices)); } catch (_) { /* Private/storage-disabled browser. */ }
}
function fillSelect(id, defs, chosen, allowNone = false) {
  const target = $(id);
  if (allowNone) target.append(new Option("None", ""));
  Object.entries(defs).forEach(([key, value]) => target.append(new Option(value.label || value, key)));
  target.value = chosen;
}
fillSelect("leftMetric", metricDefs, "episode_return");
fillSelect("rightMetric", metricDefs, "transitions_per_second", true);
fillSelect("scatterX", scatterDefs, "hp_cost");
fillSelect("scatterY", scatterDefs, "floor");
restored = applySavedOnce(restored, saved => {
  for (const id of savedIDs) if (saved[id] !== undefined) {
    if ($(id).type === "checkbox") $(id).checked = !!saved[id];
    else if ($(id).type === "number") $(id).value = saved[id];
    else {if(![...$(id).options].some(option => option.value === saved[id]))$(id).add(new Option(String(saved[id]),String(saved[id])));$(id).value=saved[id];}
  }
  if (Array.isArray(saved.range) && saved.range.length === 2 && saved.range.every(valid)) state.range = saved.range;
});
function tooltip(event, text) {
  const box = $("tooltip"); box.textContent = text; box.hidden = false;
  const left = Math.min(event.clientX + 14, window.innerWidth - Math.min(330, box.offsetWidth) - 10);
  const top = Math.min(event.clientY + 14, window.innerHeight - box.offsetHeight - 10);
  box.style.left = `${Math.max(8, left)}px`; box.style.top = `${Math.max(8, top)}px`;
}
function hideTooltip() { $("tooltip").hidden = true; }
function legend(target, series) {
  const container = el("div", undefined, "legend");
  for (const [index, line] of series.entries()) {
    const key = el("span"), dot = el("b"); dot.style.background = line.color || palette[index % palette.length];
    key.append(dot, document.createTextNode(line.label + (line.points&&!line.points.some(point=>valid(point.y))?" · unavailable":"") + (line.axis === "right" ? " · right axis" : ""))); container.append(key);
  }
  target.append(container);
}
function bounds(values, fixed = null) {
  if (fixed) return fixed;
  const good = values.filter(valid);
  if (!good.length) return [0, 1];
  let low = Math.min(...good), high = Math.max(...good);
  if (low === high) { const pad = Math.max(.01, Math.abs(low) * .1); low -= pad; high += pad; }
  const pad = (high - low) * .08;
  return [low - pad, high + pad];
}
function rangeRows(rows) { return rows.filter(row => valid(row.round) && row.round >= state.range[0] && row.round <= state.range[1]); }
function roundMetric(key) {
  const def = metricDefs[key];
  const rows = def?.episode ? perRoundEpisodes(state.episodes, $("includeCuts").checked) : state.data.training;
  return rangeRows(rows).map(row => ({x: row.round, y: valid(row[key]) ? row[key] : null,
    note: def?.episode ? `${row.n} requested; ${row.numeric_n} numeric eligible` : `Job ${row.job}; ${row.status}; ${row.session || "session unavailable"}`}));
}
function smoothSeries(key, color, axis = "left") {
  const raw = roundMetric(key), window = Number($("rolling").value), def = metricDefs[key];
  const series = [];
  if ($("rawLines").checked && window > 1) series.push({label: `${def.label} · raw`, points: raw, axis, color, opacity: .2, width: 1, hideLegend: true});
  series.push({label: `${def.label}${window > 1 ? ` · rolling ${window}` : ""}`, points: rolling(raw, window), axis, color, percent: def.percent});
  return series;
}
function plot(id, series, options = {}) {
  const target = $(id); target.replaceChildren();
  const points = series.flatMap(line => line.points).filter(p => valid(p.x) && valid(p.y));
  if (!points.length) { target.append(el("div", "No logged numeric observations for this selection.", "empty")); return; }
  const width = Math.max(180, target.clientWidth), height = options.large ? 340 : 250;
  const dual = series.some(line => line.axis === "right");
  const L = width < 320 ? 54 : 64, R = dual ? (width < 320 ? 54 : 66) : 17, T = 18, B = 39;
  const xDomain = options.xDomain || bounds(points.map(p => p.x));
  if (xDomain[0] === xDomain[1]) xDomain[1] += 1;
  const leftDomain = bounds(series.filter(line => line.axis !== "right").flatMap(line => line.points.map(p => p.y)), options.yDomain);
  const rightDomain = bounds(series.filter(line => line.axis === "right").flatMap(line => line.points.map(p => p.y)), options.rightDomain);
  const x = v => L + (v - xDomain[0]) / (xDomain[1] - xDomain[0]) * (width - L - R);
  const y = (v, axis) => { const d = axis === "right" ? rightDomain : leftDomain; return height - B - (v - d[0]) / (d[1] - d[0]) * (height - T - B); };
  const canvas = svg("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": series.filter(s=>!s.hideLegend).map(s=>s.label).join("; "), class: options.interactive ? "interactive" : ""});
  const clipID = `clip-${id}`;
  const defs = svg("defs"), clip = svg("clipPath", {id: clipID}); clip.append(svg("rect", {x:L,y:T,width:Math.max(1,width-L-R),height:height-T-B})); defs.append(clip); canvas.append(defs);
  for (let index = 0; index <= 4; index++) {
    const yy = T + (height-T-B)*index/4;
    canvas.append(svg("line", {x1:L,x2:width-R,y1:yy,y2:yy,class:"grid"}));
    const left = leftDomain[1]-(leftDomain[1]-leftDomain[0])*index/4;
    canvas.append(svg("text", {x:L-7,y:yy+4,"text-anchor":"end"}, options.percent ? pct(left) : fmt(left,3)));
    if (dual) {
      const right = rightDomain[1]-(rightDomain[1]-rightDomain[0])*index/4;
      canvas.append(svg("text", {x:width-R+7,y:yy+4,"text-anchor":"start"}, options.rightPercent ? pct(right) : fmt(right,3)));
    }
  }
  const ticks = width > 500 ? 5 : 3;
  for (let index = 0; index < ticks; index++) {
    const value = xDomain[0] + (xDomain[1]-xDomain[0])*index/(ticks-1);
    canvas.append(svg("text", {x:x(value),y:height-14,"text-anchor":"middle"}, fmt(value,0)));
  }
  const paths = svg("g", {"clip-path":`url(#${clipID})`});
  for (const [index, line] of series.entries()) {
    const color = line.color || palette[index % palette.length];
    const visible = line.points.filter(p => valid(p.x) && p.x >= xDomain[0] && p.x <= xDomain[1]);
    const step = Math.max(1, Math.ceil(visible.length/700));
    const sampled = visible.filter((p,i)=>i%step===0||i===visible.length-1);
    let path = "", connected = false;
    for (const point of sampled) {
      if (!valid(point.y)) { connected=false; continue; }
      path += `${connected ? "L" : "M"}${x(point.x)},${y(point.y,line.axis)} `; connected=true;
    }
    paths.append(svg("path", {d:path,fill:"none",stroke:color,"stroke-width":line.width||2,"stroke-opacity":line.opacity??1,"stroke-dasharray":line.dashed?"5 4":""}));
    if (!line.opacity && !line.dashed && sampled.length <= 300) for (const point of sampled.filter(p=>valid(p.y))) {
      const dot = svg("circle", {cx:x(point.x),cy:y(point.y,line.axis),r:2.5,fill:point.rejected?"white":color,stroke:color});
      dot.append(svg("title", {}, `${line.label}: ${fmt(point.y,5)} · round ${point.x}\n${point.note||""}`)); paths.append(dot);
    }
  }
  canvas.append(paths); target.append(canvas); legend(target,series.filter(line=>!line.hideLegend));
  canvas.addEventListener("pointermove", event => {
    if (state.drag) return;
    const rect=canvas.getBoundingClientRect(), mouseX=(event.clientX-rect.left)*width/rect.width;
    if (mouseX < L || mouseX > width-R) {hideTooltip();return;}
    const round=xDomain[0]+(mouseX-L)/(width-L-R)*(xDomain[1]-xDomain[0]);
    const lines=[];
    for (const line of series.filter(s=>!s.hideLegend&&!s.dashed)) {
      const candidate = line.points.filter(p=>valid(p.y)).reduce((best,p)=>!best||Math.abs(p.x-round)<Math.abs(best.x-round)?p:best,null);
      if(candidate) lines.push(`${line.label}: ${line.percent?pct(candidate.y):fmt(candidate.y,5)}\nround ${candidate.x}${candidate.contributors?` · ${candidate.contributors} contributing rounds`:""}${candidate.note?` · ${candidate.note}`:""}`);
    }
    if (lines.length) tooltip(event,lines.join("\n\n"));
  });
  canvas.addEventListener("pointerleave", hideTooltip);
  if (options.interactive) {
    canvas.addEventListener("wheel", event => {
      event.preventDefault(); const rect=canvas.getBoundingClientRect(), fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-L)/(rect.width-L-R)));
      const center=state.range[0]+fraction*(state.range[1]-state.range[0]); zoomRange(event.deltaY>0?1.25:.8,center,fraction);
    },{passive:false});
    canvas.addEventListener("pointerdown", event => {
      if(event.button!==0)return; event.preventDefault(); hideTooltip();
      state.drag={kind:"range",startX:event.clientX,range:[...state.range],pixels:Math.max(1,width-L-R)};
      $("follow").checked=false;
    });
  }
}
function overview() {
  const target=$("overview");target.replaceChildren();
  if(!state.data||!state.rangeBounds)return;
  const width=Math.max(180,target.clientWidth),height=58,[low,high]=state.rangeBounds;
  const x=value=>5+(value-low)/Math.max(1,high-low)*(width-10);
  const points=state.data.training.filter(row=>valid(row.round)&&valid(row.return_mean));
  const domain=bounds(points.map(row=>row.return_mean)), y=value=>45-(value-domain[0])/(domain[1]-domain[0])*35;
  const canvas=svg("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":"Full training history range selector"});
  canvas.append(svg("polyline",{points:points.map(row=>`${x(row.round)},${y(row.return_mean)}`).join(" "),fill:"none",stroke:"#91a8cc","stroke-width":1.3}));
  canvas.append(svg("rect",{x:x(state.range[0]),y:4,width:Math.max(2,x(state.range[1])-x(state.range[0])),height:46,fill:"#3775d51a",stroke:"#5685cb"}));
  for(const edge of state.range)canvas.append(svg("rect",{x:x(edge)-3,y:3,width:6,height:48,fill:"#5685cb",rx:2}));
  target.append(canvas);
  canvas.addEventListener("pointerdown",event=>{
    if(event.button!==0)return;event.preventDefault();const rect=canvas.getBoundingClientRect();
    const round=low+Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*(high-low);
    $("follow").checked=false;state.drag={kind:"brush",startRound:round,rect,bounds:[low,high]};setRange([round-1,round+1],false);
  });
}
function setRange(range, fetchEpisodes=true) {
  if(!state.rangeBounds)return;
  const [min,max]=state.rangeBounds;
  let low=Math.max(min,Math.min(max,Math.floor(range[0]))),high=Math.max(min,Math.min(max,Math.ceil(range[1])));
  if(high<low)[low,high]=[high,low];
  if(high-low<1&&max>min){high=Math.min(max,low+1);low=Math.max(min,high-1);}
  state.range=[low,high];$("rangeMin").value=low;$("rangeMax").value=high;
  saveChoices();renderTraining();
  if(fetchEpisodes) scheduleEpisodes();
}
function zoomRange(factor, center=null, fraction=.5) {
  const width=Math.max(2,(state.range[1]-state.range[0])*factor),anchor=center??avg(state.range);
  let low=anchor-width*fraction,high=anchor+width*(1-fraction);
  if(low<state.rangeBounds[0]){high+=state.rangeBounds[0]-low;low=state.rangeBounds[0];}
  if(high>state.rangeBounds[1]){low-=high-state.rangeBounds[1];high=state.rangeBounds[1];}
  $("follow").checked=false;setRange([low,high]);
}
function presetRange() {
  if(!state.rangeBounds)return;
  const count=$("rangePreset").value,[low,high]=state.rangeBounds;
  setRange(count==="all"?[low,high]:[Math.max(low,high-Number(count)+1),high]);
}
function renderTraining() {
  if(!state.data||!state.range)return;
  const left=$("leftMetric").value,right=$("rightMetric").value;
  let series=smoothSeries(left,palette[0]);
  if(right)series=series.concat(smoothSeries(right,palette[1],"right"));
  plot("mainChart",series,{large:true,interactive:true,xDomain:[...state.range],percent:metricDefs[left].percent,rightPercent:right&&metricDefs[right].percent});
  overview();
  plot("lossChart",smoothSeries("policy_loss",palette[0]).concat(smoothSeries("value_loss",palette[1],"right")),{interactive:true,xDomain:[...state.range]});
  plot("klChart",smoothSeries("approx_kl",palette[0]).concat(smoothSeries("entropy",palette[2],"right")),{interactive:true,xDomain:[...state.range]});
  plot("timingChart",smoothSeries("rollout_seconds",palette[0]).concat(smoothSeries("update_seconds",palette[1])),{interactive:true,xDomain:[...state.range]});
}
function filters() {
  return {kind:$("episodeKind").value,source:$("source").value,start_act:$("startAct").value,
    goal_act:$("goalAct").value,component:$("component").value,session:$("session").value,outcome:$("outcome").value,
    start_floor_min:$("startFloorMin").value,start_floor_max:$("startFloorMax").value,horizon:$("horizon").value,role:$("episodeKind").value==="full3"?$("snapshotRole").value:""};
}
function episodeQuery() { return {...filters(),campaign:selectedCampaign,...(state.data?.condition?.id==="v2"&&$("episodeKind").value!=="train"?{panel_id:$("episodePanel").value}:{}),round_min:state.range[0],round_max:state.range[1],max_jobs:1000}; }
function describeCohort() {
  const f=filters();return `${$("episodeKind").selectedOptions[0].text} · rounds ${state.range.join("–")} · ${f.source||"all sources"} · start act ${f.start_act||"all"} → goal ${f.goal_act||"all"}${f.component?` · ${f.component}`:""}${f.session?` · ${f.session}`:""}${f.outcome?` · ${f.outcome}`:""}${f.start_floor_min||f.start_floor_max?` · start floor ${f.start_floor_min||"any"}–${f.start_floor_max||"any"}`:""}${f.horizon?` · horizon ${f.horizon}`:""}${f.role?` · ${f.role}`:""}${state.data?.condition?.id==="v2"&&f.kind!=="train"?` · ${$("episodePanel").selectedOptions[0].text}`:""}`;
}
function scheduleEpisodes() {
  clearTimeout(state.interactionTimer);state.interactionTimer=setTimeout(loadEpisodes,240);
}
function episodeVersion(kind) {
  if(state.data?.condition?.id==="v2"&&kind!=="train")return JSON.stringify(state.data.v2.panels[$("episodePanel").value]?.history||[]);
  if(kind==="full3")return JSON.stringify(state.data.full_run?.history.map(row=>[row.checkpoint_sha256,row.evaluated_at,row.roles])||[]);
  if(kind==="dev")return JSON.stringify(state.data.dev_history.filter(row=>row.round>=state.range[0]&&row.round<=state.range[1]).map(row=>[row.round,row.at]));
  return JSON.stringify(rangeRows(state.data.training).filter(row=>row.status==="complete").map(row=>[row.job,row.round,row.finished]));
}
async function loadEpisodes(force=false) {
  if(!state.data||!state.range)return;
  const query=episodeQuery(),key=JSON.stringify(query)+":"+episodeVersion(query.kind);
  if(!force&&(key===state.episodeKey||(state.episodeLoading&&key===state.inFlightEpisodeKey)))return;
  state.episodeAbort?.abort();const controller=new AbortController();state.episodeAbort=controller;
  const token=++state.episodeToken;state.episodeKey=null;state.inFlightEpisodeKey=key;state.episodeLoading=true;state.episodeError=null;state.episodes=[];state.episodeResponse=null;
  renderEpisodes();renderTraining();
  const rows=[];let responseData=null,datasetID=null;
  try {
    while(rows.length<50000){
      const params=new URLSearchParams(Object.entries(query).filter(([,v])=>v!==""&&v!==null).map(([k,v])=>[k,String(v)]));
      params.set("limit",String(Math.min(10000,50000-rows.length)));params.set("offset",String(rows.length));
      const response=await fetch(`/api/episodes?${params}`,{signal:controller.signal,cache:"no-store"});
      if(!response.ok)throw new Error(`Episode API HTTP ${response.status}`);
      const page=await response.json();if(!Array.isArray(page.episodes))throw new Error("Episode API returned no episode array");
      if(typeof page.dataset_id!=="string")throw new Error("Episode API has no stable dataset identity; refresh after backend update.");
      if(datasetID!==null&&page.dataset_id!==datasetID)throw new Error("Data changed while loading pages; mixed pages discarded. Refresh to retry.");datasetID=page.dataset_id;
      rows.push(...page.episodes);responseData=page;
      if(token!==state.episodeToken)return;
      $("cohort").textContent=`Loading ${fmt(rows.length,0)} of ${fmt(page.total,0)} filtered records · ${describeCohort()}`;
      if(!page.has_more||!page.episodes.length)break;
    }
    if(token!==state.episodeToken)return;
    state.episodes=rows;state.episodeResponse=responseData;state.episodeKey=key;state.scatterDomain=null;
    updateAvailability(rows,responseData?.availability);
  } catch(error){if(error.name!=="AbortError"&&token===state.episodeToken)state.episodeError=error.message;}
  finally{if(token===state.episodeToken){state.inFlightEpisodeKey=null;state.episodeLoading=false;renderEpisodes();renderTraining();}}
}
function updateAvailability(rows, availability) {
  const mappings={source:"start_source",component:"component",session:"session",outcome:"outcome",horizon:"horizon"};
  for(const [id,key]of Object.entries(mappings)){
    const select=$(id),chosen=select.value;
    const extra=Array.isArray(availability?.filter_options?.[key])?availability.filter_options[key]:[];
    const values=new Set([...rows.map(row=>row[key]).filter(v=>v!==null&&v!==undefined),...extra]);
    for(const value of values)if(![...select.options].some(option=>option.value===String(value)))select.add(new Option(String(value),String(value)));
    select.value=chosen;
  }
}
function renderHistogram(id, rows, field, color) {
  const target=$(id);target.replaceChildren();
  const data=histogram(rows.map(row=>row[field]),Number($("bins").value));
  if(!data.count){target.append(el("div","No logged numeric values in this eligible cohort.","empty"));return;}
  const width=Math.max(180,target.clientWidth),height=250,L=49,R=15,T=18,B=41;
  const max=Math.max(1,...data.bins.map(bin=>bin.count));
  const canvas=svg("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":`${field} histogram of ${data.count} eligible measurements`});
  for(let i=0;i<=4;i++){
    const yy=T+(height-T-B)*i/4;
    canvas.append(svg("line",{x1:L,x2:width-R,y1:yy,y2:yy,class:"grid"}),svg("text",{x:L-7,y:yy+4,"text-anchor":"end"},fmt(max*(1-i/4),0)));
  }
  const barWidth=(width-L-R)/data.bins.length;
  data.bins.forEach((bin,i)=>{
    const barHeight=bin.count/max*(height-T-B);
    const bar=svg("rect",{x:L+i*barWidth+.5,y:height-B-barHeight,width:Math.max(.5,barWidth-1),height:barHeight,fill:color,opacity:.8});
    bar.addEventListener("pointermove",event=>tooltip(event,`${fmt(bin.low,3)} → ${fmt(bin.high,3)}\n${fmt(bin.count,0)} observations · ${pct(bin.count/data.count)} of numeric measurements`));
    bar.addEventListener("pointerleave",hideTooltip);canvas.append(bar);
  });
  const first=data.bins[0].low,last=data.bins.at(-1).high;
  [first,(first+last)/2,last].forEach((value,i)=>canvas.append(svg("text",{x:L+i*(width-L-R)/2,y:height-15,"text-anchor":"middle"},fmt(value,2))));
  target.append(canvas,el("div",`${fmt(data.count,0)} measured · ${fmt(data.missing,0)} unavailable among eligible records · median ${fmt(percentile(rows.map(row=>row[field]),.5))}`,"legend"));
}
function renderFamilyChart() {
  const target=$("familyChart");target.replaceChildren();
  if(!state.episodes.length){target.append(el("div","Load a cohort to compare requested outcome families.","empty"));return;}
  const groups=new Map();
  for(const row of state.episodes){
    if(row.requested_occurrence===false)continue;
    const key=`${row.start_source||"unknown source"} · ${row.start_act??"?"}→${row.goal_act??"?"}`;
    if(!groups.has(key))groups.set(key,{n:0,counts:{}});
    const group=groups.get(key);group.n++;const outcome=outcomeClass(row);group.counts[outcome]=(group.counts[outcome]||0)+1;
  }
  const entries=[...groups].sort((a,b)=>b[1].n-a[1].n).slice(0,15),width=Math.max(180,target.clientWidth),height=Math.max(250,entries.length*33+42);
  const canvas=svg("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":"Requested outcome fractions by start source and act goal"});
  canvas.style.height=`${height}px`;
  const L=Math.min(190,width*.47),R=10,barWidth=width-L-R;
  entries.forEach(([key,group],i)=>{
    const yy=17+i*33;canvas.append(svg("text",{x:5,y:yy+13},key.length>28?key.slice(0,26)+"…":key));
    let start=0;
    for(const outcome of ["success","defeat","fault","cutoff","unstarted","other"]){
      const count=group.counts[outcome]||0;if(!count)continue;
      const bar=svg("rect",{x:L+start/group.n*barWidth,y:yy,width:count/group.n*barWidth,height:20,fill:outcomeColors[outcome]});
      bar.addEventListener("pointermove",event=>tooltip(event,`${key}\n${outcome}: ${count}/${group.n} requested · ${pct(count/group.n)}\nIncludes faults, cutoffs, missing/unstarted outcomes in denominator.`));bar.addEventListener("pointerleave",hideTooltip);canvas.append(bar);start+=count;
    }
  });
  target.append(canvas);legend(target,Object.entries(outcomeColors).map(([label,color])=>({label,color})));
}
function renderCoverageChart(numeric) {
  const target=$("coverageChart");target.replaceChildren();
  if(!state.episodes.some(row=>row.requested_occurrence!==false)){target.append(el("div","No loaded requested episode coverage.","empty"));return;}
  const items=[["Return",numeric.filter(row=>valid(row.reward)).length],["Ending floor",numeric.filter(row=>valid(row.floor)).length],["Net HP cost",numeric.filter(row=>valid(row.hp_cost)).length],["Steps",numeric.filter(row=>valid(row.steps)).length]];
  const width=Math.max(180,target.clientWidth),height=250,L=95,R=20,total=state.episodes.filter(row=>row.requested_occurrence!==false).length;
  const canvas=svg("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":"Measured versus missing or excluded records"});
  items.forEach(([label,count],i)=>{
    const yy=30+i*44;canvas.append(svg("text",{x:5,y:yy+15},label));
    canvas.append(svg("rect",{x:L,y:yy,width:width-L-R,height:23,fill:"#e5ecf5",rx:3}));
    const bar=svg("rect",{x:L,y:yy,width:count/total*(width-L-R),height:23,fill:palette[i],rx:3});
    bar.addEventListener("pointermove",event=>tooltip(event,`${label}: ${count}/${total} loaded requested records measured and eligible\n${total-count} unavailable or excluded (fault/cutoff/missing).`));bar.addEventListener("pointerleave",hideTooltip);canvas.append(bar);
    canvas.append(svg("text",{x:L,y:yy+36},`${fmt(count,0)}/${fmt(total,0)} · ${pct(count/total)}`));
  });target.append(canvas);
}
function renderEpisodes() {
  if(!state.range)return;
  const meta=state.episodeResponse,coverage=meta?.coverage||{},rows=state.episodes;
  const warning=state.episodeError||((meta?.total||0)>rows.length?`CAPPED SAMPLE: ${fmt(rows.length,0)} of ${fmt(meta.total,0)} filtered records loaded. Narrow the range to load the entire cohort.`:"");
  $("cohort").className=`cohort${warning?" warn":""}`;
  $("cohort").textContent=state.episodeLoading?`Loading episode pages · ${describeCohort()}`:state.episodeError?`${state.episodeError} · global PPO plots remain available.`:`${describeCohort()} · ${fmt(rows.length,0)} loaded / ${fmt(meta?.total,0)} filtered in bounded window · ${fmt(coverage.scanned_jobs,0)} scanned jobs / ${fmt(coverage.eligible_jobs,0)} eligible · ${fmt(coverage.omitted_jobs||0,0)} jobs omitted · ${fmt(coverage.unjoined_rows||0,0)} records lack start metadata · ${fmt(meta?.availability?.unexpected_observed_rows||0,0)} unexpected diagnostics · ${fmt(meta?.availability?.invalid_coverage_rows||0,0)} rows excluded for invalid coverage.${$("outcome").value?" Outcome-filtered denominator; this is conditional on the chosen outcome.":""}${warning?" "+warning:""}`;
  const requested=rows.filter(row=>row.requested_occurrence!==false);
  const unexpected=rows.length-requested.length;
  const numeric=rows.filter(row=>numericEligible(row,$("includeCuts").checked));
  const counts=Object.fromEntries(Object.keys(outcomeColors).map(key=>[key,requested.filter(row=>outcomeClass(row)===key).length]));
  cards("sampleCards",[["Loaded requested records",fmt(requested.length,0),`${unexpected} unexpected diagnostic records excluded · ${fmt(numeric.length,0)} numeric eligible · ${$("includeCuts").checked?"cutoff values included":"fault/cutoff values excluded"}`],["Target successes",`${counts.success}/${requested.length}`,`${pct(requested.length?counts.success/requested.length:null)} · conditional continuation/fresh goal mixture`],["Faults / cutoffs",`${counts.fault} / ${counts.cutoff}`,`${counts.unstarted} unstarted · ${counts.other} other/missing`],["Net HP cost · measured mean",fmt(avg(numeric.map(row=>row.hp_cost))),"Reset HP minus ending HP; negative = net healing"]]);
  renderHistogram("returnHist",numeric,"reward",palette[0]);renderHistogram("floorHist",numeric,$("floorHistMetric").value,palette[2]);renderHistogram("hpHist",numeric,"hp_cost",palette[1]);renderHistogram("stepsHist",numeric,"steps",palette[3]);renderHistogram("healthHist",numeric,$("healthMetric").value,palette[2]);renderHistogram("actHist",numeric,"max_act_reached",palette[4]);
  renderFamilyChart();renderCoverageChart(numeric);renderScatter();
  $("episodeTableNote").textContent=`Latest 150 of ${fmt(rows.length,0)} loaded records. CSV contains every loaded row, including fault/cutoff/unstarted records; ${meta?.has_more?"sample capped or paging incomplete":"all filtered rows in the bounded scanned window loaded"}.`;
  const recent=rows.slice().sort((a,b)=>(b.round??-1)-(a.round??-1)).slice(0,150);
  table("episodeTable",["Round / world","Source · start → goal","Snapshot role","Outcome","Return","Ending floor","Net HP cost","Steps","Fault / cut"],recent.map(row=>[`${row.round??"—"} / ${row.world_seed??"—"}`,`${row.start_source||"unknown"} · act ${row.start_act??"?"} / floor ${row.start_floor??"?"} → goal ${row.goal_act??"?"}`,Array.isArray(row.roles)?row.roles.join(" + "):row.role||"—",row.outcome,fmt(row.reward,3),fmt(row.floor,0),fmt(row.hp_cost,2),fmt(row.steps,0),`${!!row.fault} / ${!!(row.truncated||row.censored)}`]));
  $("downloadCSV").disabled=!rows.length;
}
function renderScatter() {
  const target=$("scatter");target.replaceChildren();
  const xKey=$("scatterX").value,yKey=$("scatterY").value,groupKey=$("scatterGroup").value;
  const eligible=state.episodes.filter(row=>numericEligible(row,$("includeCuts").checked)&&valid(row[xKey])&&valid(row[yKey]));
  if(!eligible.length){target.append(el("div","No paired logged measurements for this selection.","empty"));$("scatterNote").textContent="";return;}
  const step=Math.max(1,Math.ceil(eligible.length/1800)),sampled=eligible.filter((_,i)=>i%step===0);
  $("scatterNote").textContent=`${fmt(sampled.length,0)} plotted / ${fmt(eligible.length,0)} paired measurements · deterministic stride sample · drag / wheel`;
  if(!state.scatterDomain)state.scatterDomain={x:bounds(eligible.map(row=>row[xKey])),y:bounds(eligible.map(row=>row[yKey]))};
  const width=Math.max(180,target.clientWidth),height=340,L=65,R=20,T=18,B=43,domain=state.scatterDomain;
  const x=value=>L+(value-domain.x[0])/(domain.x[1]-domain.x[0])*(width-L-R);
  const y=value=>height-B-(value-domain.y[0])/(domain.y[1]-domain.y[0])*(height-T-B);
  const groups=[...new Set(eligible.map(row=>String(row[groupKey]??"unknown")))].sort();
  const canvas=svg("svg",{viewBox:`0 0 ${width} ${height}`,class:"interactive",role:"img","aria-label":`${scatterDefs[xKey]} versus ${scatterDefs[yKey]} colored by ${groupKey}`});
  for(let i=0;i<=4;i++){
    const fraction=i/4,xx=L+fraction*(width-L-R),yy=T+fraction*(height-T-B);
    canvas.append(svg("line",{x1:L,x2:width-R,y1:yy,y2:yy,class:"grid"}),svg("text",{x:L-7,y:yy+4,"text-anchor":"end"},fmt(domain.y[1]-fraction*(domain.y[1]-domain.y[0]),2)));
    if(width>350||i%2===0)canvas.append(svg("text",{x:xx,y:height-15,"text-anchor":"middle"},fmt(domain.x[0]+fraction*(domain.x[1]-domain.x[0]),2)));
  }
  state.scatterPoints=[];
  for(const row of sampled){
    const xx=x(row[xKey]),yy=y(row[yKey]);if(xx<L||xx>width-R||yy<T||yy>height-B)continue;
    const group=String(row[groupKey]??"unknown"),index=groups.indexOf(group);
    canvas.append(svg("circle",{cx:xx,cy:yy,r:2.7,fill:palette[index%palette.length],opacity:.42}));state.scatterPoints.push({x:xx,y:yy,row});
  }
  target.append(canvas);legend(target,groups.map((label,i)=>({label,color:palette[i%palette.length]})));
  canvas.addEventListener("pointermove",event=>{
    if(state.drag)return;const rect=canvas.getBoundingClientRect(),xx=(event.clientX-rect.left)*width/rect.width,yy=(event.clientY-rect.top)*height/rect.height;
    let nearest=null,distance=500;
    for(const point of state.scatterPoints){const d=(point.x-xx)**2+(point.y-yy)**2;if(d<distance){distance=d;nearest=point.row;}}
    if(!nearest){hideTooltip();return;}
    tooltip(event,`${scatterDefs[xKey]}: ${fmt(nearest[xKey],4)}\n${scatterDefs[yKey]}: ${fmt(nearest[yKey],4)}\nRound ${nearest.round??"—"} · world ${nearest.world_seed??"—"}\n${nearest.start_source||"unknown source"} · act ${nearest.start_act??"?"} / floor ${nearest.start_floor??"?"} → goal ${nearest.goal_act??"?"}\n${nearest.outcome||"unknown outcome"}${nearest.roles?" · "+nearest.roles.join(" + "):""}${nearest.checkpoint_sha256?"\nCP "+nearest.checkpoint_sha256:""}`);
  });canvas.addEventListener("pointerleave",hideTooltip);
  canvas.addEventListener("wheel",event=>{
    event.preventDefault();const rect=canvas.getBoundingClientRect(),fx=Math.max(0,Math.min(1,((event.clientX-rect.left)*width/rect.width-L)/(width-L-R))),fy=Math.max(0,Math.min(1,1-((event.clientY-rect.top)*height/rect.height-T)/(height-T-B))),factor=event.deltaY>0?1.2:.8;
    for(const [axis,fraction]of [["x",fx],["y",fy]]){const d=state.scatterDomain[axis],center=d[0]+fraction*(d[1]-d[0]),span=(d[1]-d[0])*factor;state.scatterDomain[axis]=[center-fraction*span,center+(1-fraction)*span];}renderScatter();
  },{passive:false});
  canvas.addEventListener("pointerdown",event=>{if(event.button!==0)return;event.preventDefault();hideTooltip();state.drag={kind:"scatter",startX:event.clientX,startY:event.clientY,domain:structuredClone(state.scatterDomain),width:width-L-R,height:height-T-B};});
}
function fullRunSeries(field) {
  const full=state.data.full_run,history=full?.history||[],max=Math.max(1,state.data.global_round||1,...history.map(row=>row.captured_global_round||0));
  const initial=full?.initial,baseline=field==="clear_rate"?initial?.clears/initial?.requested:initial?.floor_mean;
  const series=[{label:"INITIAL · fixed pre-update reference",points:[{x:0,y:baseline},{x:max,y:baseline}],color:"#7c8a9f",dashed:true}];
  ["selected_best","latest_candidate"].forEach((role,index)=>series.push({label:role==="selected_best"?"Selected best · evaluated":"Latest candidate · evaluated",color:palette[index],percent:field==="clear_rate",points:history.filter(row=>row.roles.includes(role)&&row.complete).map(row=>({x:row.captured_global_round,y:row[field],note:`${row.clears}/${row.requested} wins; ${row.defeats} defeats; ${row.faults} faults / ${row.truncated} cutoffs\n${utc(row.evaluated_at)}\nSHA ${row.checkpoint_sha256}`}))}));
  return series;
}
function renderFullRuns() {
  if(state.data.v2){renderV2Performance();return;}
  $("learningCards").replaceChildren();$("performancePanelLabel").hidden=true;$("performanceScope").textContent="v1 · fixed 48-world observer · monitoring only";
  const full=state.data.full_run;if(!full)return;const initial=full.initial;
  const values=[["INITIAL · pre-update",`${initial.clears}/${initial.requested} wins`,`${initial.reference_date} · floor ${fmt(initial.floor_mean)} · CP ${shortSHA(initial.checkpoint_sha256)}`]];
  for(const role of ["selected_best","latest_candidate"]){
    const row=full.latest[role],label=role==="selected_best"?"SELECTED BEST · last evaluated":"LATEST CANDIDATE · last evaluated";
    values.push(row?[label,`${row.clears}/${row.requested} wins · floor ${fmt(row.floor_mean)}`,`${utc(row.evaluated_at)} · ${age(row.evaluation_age_seconds)} old · ${row.complete?"complete":"INCOMPLETE"} · ${row.faults} faults / ${row.truncated} cuts · CP ${shortSHA(row.checkpoint_sha256)} · ${row.matches_live_role[role]?"matches live role":"newer live SHA unevaluated"}${row.reused_evaluation?" · reused same-SHA result":""}`]:[label,"NOT YET EVALUATED","No measured panel for this role"]);
  }cards("fullcards",values);
  $("fullstatus").textContent=`Observer ${full.status} · ${full.next_due?"next check "+utc(full.next_due):"first cycle pending"} · ${full.errors.join(" ")||"observations do not change learner selection"}`;
  plot("fullwinchart",fullRunSeries("clear_rate"),{percent:true,yDomain:[0,Math.max(.05,...full.history.map(row=>valid(row.clear_rate)?row.clear_rate*1.2:0))]});
  plot("fullfloorchart",fullRunSeries("floor_mean"));
  const rows=[["INITIAL reference",initial.saved_metrics_at?utc(initial.saved_metrics_at)+" (saved)":initial.reference_date,"Before updates",`${initial.clears}/${initial.requested}`,48,"0 / 0 / 0","Complete · 48 requested",fmt(initial.floor_mean),initial.checkpoint_sha256]];
  for(const row of full.history.slice(-40).reverse())rows.push([row.roles.join(" + "),utc(row.evaluated_at),`${row.captured_global_round??"—"} · ${utc(row.capture_at)}`,`${row.clears}/${row.requested}`,row.defeats,`${row.faults} / ${row.truncated} / ${row.unstarted}`,`${row.complete?"Complete":"INCOMPLETE"} · ${row.task_count} IDs · ${row.matched_panel===true?"exact cohort":"identity unchecked/mismatched"}`,fmt(row.floor_mean),row.checkpoint_sha256]);
  table("fullhistory",["Snapshot role","Evaluated UTC","Captured round / UTC","Wins / requested","Defeats","Faults / cuts / unstarted","Coverage","Mean floor","Exact SHA256"],rows);
}
function devSeries(field) {
  const data=state.data,max=Math.max(1,data.global_round||1),series=[];
  for(const act of [1,2,3]){
    const points=[];
    for(const event of data.dev_history){const goal=event.goal_act??((event.stage||"").includes("act2")?2:(event.stage||"").includes("3")?3:null),quality=act===1?event.retention:goal===act?event.quality:null;
      if(quality&&valid(event.round))points.push({x:event.round,y:quality[field],rejected:event.decision?.action==="rollback",note:`${event.decision?.action} · ${quality.clears}/${quality.requested} · ${quality.faults} faults / ${quality.truncated} cuts · ${quality.complete?"complete":"INCOMPLETE"}`});}
    if(points.length)series.push({label:`Act ${act}${act===1?" retention":""}`,points,color:palette[act-1],percent:field==="clear_rate"});
    const baseline=data.baseline[`dev_act${act}`];series.push({label:`Act ${act} initial reference`,points:[{x:0,y:field==="clear_rate"?baseline.clears/baseline.requested:baseline.floor_mean},{x:max,y:field==="clear_rate"?baseline.clears/baseline.requested:baseline.floor_mean}],color:palette[act-1],dashed:true});
  }return series;
}
function renderAudits() {
  const data=state.data,total=data.totals,allocation=data.allocation;
  $("subtitle").textContent=`${data.condition?.id?.toUpperCase()||""} · ${data.status.toUpperCase()} · ${data.stage||"stage unavailable"} · controller round ${data.global_round??"—"}`;
  $("refreshed").textContent=`${time(data.served_at)} · live every 8s`;
  const warnings=[...data.errors,...(data.v2?.alerts||[])];if(data.heartbeat_age_seconds>45)warnings.push(`Controller heartbeat ${age(data.heartbeat_age_seconds)} old.`);if(data.pause_reason)warnings.push(data.pause_reason);$("banner").textContent=warnings.join(" ");
  cards("cards",[["Active remaining",hours(data.active_remaining_seconds),`${hours(data.active_used_seconds)} used / ${hours(allocation.authorized_seconds)} authorized`],["Valid transitions",fmt(data.committed_totals.transitions,0),`${fmt(data.committed_totals.steps,0)} committed accepted optimizer steps`],["Campaign episodes",fmt(total.episodes,0),`${total.faults} faults · ${total.truncated} truncated`],["Checkpoint age",age(data.checkpoint_age_seconds),`Heartbeat ${age(data.heartbeat_age_seconds)} old`]]);
  $("budget").value=allocation.authorized_seconds?data.active_used_seconds/allocation.authorized_seconds:0;
  details("allocation",[["Active ledger",hours(data.active_used_seconds)],["Wall elapsed",hours(data.wall_elapsed_seconds)],["Authorized",hours(allocation.authorized_seconds)],["Active job",`${data.active_job??"—"} · ${data.active_job_kind||"—"} · ${data.active_job_status||"—"}`]]);
  table("checkpoints",["Identity","Filename","Published SHA256"],Object.entries(data.checkpoints).map(([key,value])=>[key.replaceAll("_"," "),value.name,el("span",value.sha256||"Unavailable","sha")]));
  details("settings",Object.entries(data.config).map(([key,value])=>[key.replaceAll("_"," "),value===null?"Unavailable":String(value)]));
  const latest=data.training.at(-1);if(latest)details("latest",[["Round / job",`${latest.round??"unavailable"} / ${latest.job}`],["Accepted optimizer steps",fmt(latest.steps,0)],["Policy / value loss",`${fmt(latest.policy_loss,6)} / ${fmt(latest.value_loss,6)}`],["KL / entropy",`${fmt(latest.approx_kl,5)} / ${fmt(latest.entropy,4)}`],["Clip fraction",pct(latest.clip_fraction)],["Gradient norm",fmt(latest.gradient_norm)],["Rollout / update seconds",`${fmt(latest.rollout_seconds)} / ${fmt(latest.update_seconds)}`],["Rejected minibatches",fmt(latest.rejected_minibatches,0)],["Exact requested coverage",String(latest.coverage.exact_match)],["Actor changed",String(latest.actor_changed)]]);
  $("coverage").textContent=`${data.history_jobs} logged jobs · ${total.excluded_transitions} excluded transitions · ${total.unstarted} unstarted · ${data.pruning.deleted} owned artifacts pruned. Headline counts use committed journal evidence; metric history can include partial output.`;
  table("jobs",["Round / job","Session","Transitions","Accepted steps","Return mean","Rollout / update s","Faults / cuts"],data.training.slice(-30).reverse().map(row=>[`${row.round??"—"} / ${row.job}`,row.session,fmt(row.transitions,0),fmt(row.steps,0),fmt(row.return_mean,3),`${fmt(row.rollout_seconds)} / ${fmt(row.update_seconds)}`,`${row.faults} / ${row.truncated}`]));
  table("families",["Start source","Start → goal","Requested / valid episodes","Clears / requested","Defeats","Faults / cuts / unstarted"],data.families.map(row=>[row.metadata.start_source,`${row.metadata.start_act} → ${row.metadata.goal_act}`,`${row.requested} / ${row.episodes}`,`${row.clears} / ${row.requested} (${pct(row.clear_rate)})`,row.defeats,`${row.faults} / ${row.truncated} / ${row.unstarted}`]));
  table("decisions",["Round / time","Stage","Decision","Primary clears / floor","Act 1 retention","Primary faults / cuts"],data.decisions.slice(-30).reverse().map(event=>[`${event.round??"—"} · ${time(event.at)}`,event.stage,`${event.decision?.action||"event"} · ${event.decision?.reason||"—"}`,event.quality?`${event.quality.clears}/${event.quality.requested} · ${fmt(event.quality.floor_mean)} · ${event.quality.complete?"complete":"INCOMPLETE"}`:"—",event.retention?`${event.retention.clears}/${event.retention.requested}`:"—",event.quality?`${event.quality.faults} / ${event.quality.truncated}`:"—"]));
  $("development").hidden=!!data.v2;
  if(data.v2)return;
  const devCards=[1,2,3].map(act=>{const quality=data.leaders[`dev_act${act}`]?.quality,baseline=data.baseline[`dev_act${act}`];return [`Act ${act}${act===1?" retention anchor":act===3?" INITIAL reference":" curriculum-selected"}`,quality?`${quality.clears}/${quality.requested}`:"Unavailable",`Initial ${baseline.clears}/48 · initial floor ${fmt(baseline.floor_mean)}${quality?" · leader floor "+fmt(quality.floor_mean):""}`];});
  if(data.selected_dev?.retention)devCards.push(["Selected primary CP · paired Act 1",`${data.selected_dev.retention.clears}/${data.selected_dev.retention.requested}`,`Round ${data.selected_dev.round} · ${data.selected_dev.retention.faults} faults / ${data.selected_dev.retention.truncated} cuts`]);
  cards("devcards",devCards);plot("devchart",devSeries("clear_rate"),{percent:true,yDomain:[0,1]});plot("floorchart",devSeries("floor_mean"));
}
function render(data) {
  state.data=data;configureCondition(data);$("snapshotRole").disabled=$("episodeKind").value!=="full3";
  const rounds=data.training.map(row=>row.round).filter(valid);
  if(!rounds.length){state.rangeBounds=[0,Math.max(1,data.global_round||0)];state.range=state.range||[...state.rangeBounds];$("rangeMin").value=state.range[0];$("rangeMax").value=state.range[1];renderAudits();renderFullRuns();renderTraining();scheduleEpisodes();return;}
  const oldBounds=state.rangeBounds;state.rangeBounds=[Math.min(...rounds),Math.max(...rounds)];
  $("rangeMin").min=$("rangeMax").min=state.rangeBounds[0];$("rangeMin").max=$("rangeMax").max=state.rangeBounds[1];
  if(!state.range) {
    const preset=$("rangePreset").value;state.range=preset==="all"?[...state.rangeBounds]:[Math.max(state.rangeBounds[0],state.rangeBounds[1]-Number(preset)+1),state.rangeBounds[1]];
  }else if($("follow").checked){
    const span=state.range[1]-state.range[0];state.range=[Math.max(state.rangeBounds[0],state.rangeBounds[1]-span),state.rangeBounds[1]];
  }else state.range=[Math.max(state.rangeBounds[0],state.range[0]),Math.min(state.rangeBounds[1],state.range[1])];
  $("rangeMin").value=state.range[0];$("rangeMax").value=state.range[1];
  renderAudits();renderFullRuns();renderTraining();scheduleEpisodes();saveChoices();
}
async function refresh() {
  if(state.loading)return;const requestedCampaign=selectedCampaign,token=(state.telemetryToken||0)+1;state.telemetryToken=token;state.loading=true;$("refresh").disabled=true;
  try{const response=await fetch(`/api/telemetry?campaign=${selectedCampaign}`,{cache:"no-store"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const data=await response.json();if(token===state.telemetryToken&&requestedCampaign===selectedCampaign)render(data);}
  catch(error){if(token===state.telemetryToken)$("banner").textContent=`Telemetry polling failed: ${error.message}. Last successful data remains displayed.`;}
  finally{if(token===state.telemetryToken){state.loading=false;$("refresh").disabled=false;}}
}
function downloadCSV() {
  const fields=["kind","round","job","session","stage","world_seed","task_id","condition_id","metadata_joined","start_source","start_act","start_floor","goal_act","horizon","component","outcome","fault","truncated","censored","terminated","reward","floor","end_act","hp_cost","steps","floor_progress","start_hp","start_max_hp","end_hp","end_max_hp","end_hp_fraction","max_act_reached","termination_phase","termination_room","panel_id","roles","checkpoint_sha256","requested_occurrence","coverage_valid","numeric_eligible"];
  const cell=value=>{if(value===null||value===undefined)return "";let text=Array.isArray(value)?value.join("|"):String(value);if(typeof value==="string"&&/^[=+@-]/.test(text))text="'"+text;return /[,"\r\n]/.test(text)?'"'+text.replaceAll('"','""')+'"':text;};
  const content=fields.join(",")+"\n"+state.episodes.map(row=>fields.map(field=>cell(row[field])).join(",")).join("\n");
  const url=URL.createObjectURL(new Blob([content],{type:"text/csv;charset=utf-8"})),link=el("a");link.href=url;link.download=`rl-${$("episodeKind").value}-rounds-${state.range.join("-")}-loaded-${state.episodes.length}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function configureCondition(data){
  const v2=!!data.v2;
  $("performancePanelLabel").hidden=!v2;$("episodePanelLabel").hidden=!v2;
  const current=$("snapshotRole").value,roles=v2?{initial_anchor:"INITIAL anchor",learner:"Continuous learner",champion:"Evaluation champion"}:{initial_anchor:"INITIAL anchor",selected_best:"Selected best",latest_candidate:"Latest candidate"};
  $("snapshotRole").replaceChildren(new Option("All snapshot roles",""),...Object.entries(roles).map(([key,label])=>new Option(label,key)));
  $("snapshotRole").value=Object.hasOwn(roles,current)?current:"";
  $("episodeKind").options[2].text=v2?"Full-game DEV panels":"Full-run observer DEV";
  $("episodeKind").options[1].text=v2?"Act 1 retention DEV":"Curriculum DEV";
}
function renderV2Performance(){
  const v2=state.data.v2,panelID=$("performancePanel").value,panel=v2.panels[panelID],count=panel.requested;
  $("performancePanelLabel").hidden=false;$("performanceScope").textContent=`${panelID} · ${count} separate requested worlds · champion selected on confirmation`;
  const initial=panel.initial,latest=panel.history.filter(row=>row.role==="learner").at(-1),championSHA=v2.champion.sha256;
  const championRecord=panel.history.filter(row=>row.checkpoint_sha256===championSHA).at(-1);
  const championQuality=championRecord?.quality||(panelID==="confirm_full3"?v2.champion.quality:panelID==="screen_full3"?v2.champion.screen_quality:v2.champion.retention);
  const card=(label,quality,sha,note)=>[label,quality?`${quality.clears}/${quality.requested} ${panelID==="retention_act1"?"clears":"wins"} · floor ${fmt(quality.floor_mean)}`:"NOT YET EVALUATED",`CP ${shortSHA(sha)} · ${note}${quality?` · ${quality.faults} faults / ${quality.truncated} cuts`:""}`];
  cards("fullcards",[card("INITIAL · original 57f1",initial,v2.initial_checkpoint_sha256,`${count}-world ${panelID} baseline`),card("LEARNER · last evaluated",latest?.quality,latest?.checkpoint_sha256,latest?`${utc(latest.evaluated_at)} · ${latest.matches_live_role?"matches live learner":"newer learner SHA unevaluated"}`:"No learner observation"),card("CHAMPION · retained evaluation best",championQuality,championSHA,championRecord?utc(championRecord.evaluated_at):"Initial anchor or no observation")]);
  const learning=v2.learning,transfer=v2.allocation_transfer;
  cards("learningCards",[["Updates retained · learner",fmt(learning.learner_lineage.updates,0),`${fmt(learning.learner_lineage.chunks,0)} verified lineage chunks · ${fmt(learning.discarded_updates,0)} updates discarded`],["Updates retained · champion",fmt(learning.champion_lineage.updates,0),"Evaluation champion is separate from the continuing learner"],["Cold Adam / warm restores",`${fmt(learning.cold_optimizer_count,0)} / ${fmt(learning.optimizer_restore_count,0)}`,`${fmt(learning.rollback_count,0)} learner rollbacks`],["Active time since champion gain",hours(learning.plateau_active_seconds),`${pct(learning.evaluation_overhead_fraction)} recent child runtime in evaluation`]]);
  $("fullstatus").textContent=`Continuous learner proceeds through ordinary DEV regressions; champion selection uses its separate gates.${transfer?` Original allocation ${hours(transfer.original_allocation_seconds)} · v1 used ${hours(transfer.v1_used_seconds)} · transferred ${hours(transfer.transferred_seconds)} including pilot.`:""}`;
  const max=Math.max(1,state.data.global_round||1),history=panel.history.filter(row=>row.complete&&valid(row.round));
  function series(field){
    const value=initial?(field==="clear_rate"?initial.clear_rate:initial.floor_mean):null;
    const output=[{label:`INITIAL · ${count}-world ${panelID}`,points:[{x:0,y:value},{x:max,y:value}],dashed:true,color:"#7c8a9f"}];
    for(const [role,color]of [["learner",palette[0]],["champion",palette[1]]])output.push({label:role==="learner"?"Continuous learner · evaluated":"Retained champion · evaluated",color,percent:field==="clear_rate",points:history.filter(row=>row.role===role||(role==="champion"&&(row.decision==="promote"||row.decision?.action==="promote"))).map(row=>({x:row.round,y:row.quality?.[field],note:`${count}-world ${row.panel_id} · ${row.quality.clears}/${row.quality.requested}; ${row.quality.faults} faults / ${row.quality.truncated} cuts
${utc(row.evaluated_at)}
SHA ${row.checkpoint_sha256}`}))});return output;
  }
  plot("fullwinchart",series("clear_rate"),{percent:true,yDomain:[0,Math.max(.05,...history.map(row=>row.quality.clear_rate*1.2))]});plot("fullfloorchart",series("floor_mean"));
  const rows=[];if(initial)rows.push(["INITIAL anchor",panelID,`${initial.clears}/${initial.requested}`,fmt(initial.floor_mean),`${initial.faults}/${initial.truncated}`,"Imported validated baseline",v2.initial_checkpoint_sha256]);
  for(const row of panel.history.slice(-40).reverse())rows.push([row.role,row.panel_id,`${row.quality?.clears??"?"}/${row.quality?.requested??"?"}`,fmt(row.quality?.floor_mean),`${row.quality?.faults??"?"}/${row.quality?.truncated??"?"}`,`${row.complete?"exact complete":"INCOMPLETE / mismatched"} · round ${row.round} · ${utc(row.evaluated_at)}`,row.checkpoint_sha256]);
  table("fullhistory",["Role","Independent panel","Wins / requested","Mean floor","Faults / cuts","Coverage / evaluation UTC","Exact SHA256"],rows);
}
$("performancePanel").addEventListener("change",()=>{saveChoices();renderFullRuns();});
$("campaign").addEventListener("change",()=>{
  saveChoices();selectedCampaign=$("campaign").value;state.telemetryToken=(state.telemetryToken||0)+1;state.loading=false;$("subtitle").textContent=`Loading ${selectedCampaign} condition…`;$("banner").textContent="Loading selected condition; previous observations are being replaced.";try{localStorage.setItem("sts-rl-condition",selectedCampaign);}catch(_){}
  state.episodeAbort?.abort();state.episodeToken++;state.episodeKey=null;state.episodeLoading=false;state.data=null;state.range=null;state.rangeBounds=null;state.episodes=[];state.episodeResponse=null;state.scatterDomain=null;
  for(const id of ["source","startAct","goalAct","component","session","outcome","startFloorMin","startFloorMax","horizon","snapshotRole"])$(id).value="";
  $("episodeKind").value="train";$("follow").checked=true;
  try{const saved=JSON.parse(localStorage.getItem(storageKey())||"null");applySavedOnce(saved,view=>{
    for(const id of savedIDs)if(view[id]!==undefined){const target=$(id);if(target.type==="checkbox")target.checked=!!view[id];else if(target.type==="number")target.value=view[id];else{if(![...target.options].some(option=>option.value===view[id]))target.add(new Option(String(view[id]),String(view[id])));target.value=view[id];}}
    if(Array.isArray(view.range)&&view.range.length===2&&view.range.every(valid))state.range=view.range;
  });}catch(_){}
  refresh();
});

$("resetChoices").addEventListener("click",()=>{try{localStorage.removeItem(storageKey());}catch(_){}location.reload();});
$("refresh").addEventListener("click",()=>{state.episodeKey=null;refresh();});
$("auto").addEventListener("change",()=>{saveChoices();if($("auto").checked)refresh();});
for(const id of ["leftMetric","rightMetric","rolling","rawLines"])$(id).addEventListener("change",()=>{saveChoices();renderTraining();});
$("rangePreset").addEventListener("change",()=>{saveChoices();presetRange();});
$("rangeMin").addEventListener("change",()=>{$("follow").checked=false;setRange([Number($("rangeMin").value),state.range[1]]);});
$("rangeMax").addEventListener("change",()=>{$("follow").checked=false;setRange([state.range[0],Number($("rangeMax").value)]);});
$("zoomIn").addEventListener("click",()=>zoomRange(.6));$("zoomOut").addEventListener("click",()=>zoomRange(1.7));
$("resetRange").addEventListener("click",()=>{$("follow").checked=true;presetRange();});
$("follow").addEventListener("change",()=>{saveChoices();if($("follow").checked)presetRange();});
for(const id of ["episodeKind","source","startAct","goalAct","component","session","outcome","startFloorMin","startFloorMax","horizon","snapshotRole","episodePanel"])$(id).addEventListener("change",()=>{if(state.data?.v2){if(id==="episodeKind"&&$("episodeKind").value==="dev")$("episodePanel").value="retention_act1";if(id==="episodeKind"&&$("episodeKind").value==="full3"&&$("episodePanel").value==="retention_act1")$("episodePanel").value="confirm_full3";if(id==="episodePanel"&&$("episodeKind").value!=="train")$("episodeKind").value=$("episodePanel").value==="retention_act1"?"dev":"full3";}
  saveChoices();state.scatterDomain=null;$("snapshotRole").disabled=$("episodeKind").value!=="full3";if(id==="episodeKind"&&$(id).value==="full3"){$("follow").checked=false;setRange([0,state.rangeBounds[1]]);}else scheduleEpisodes();});
$("clearFilters").addEventListener("click",()=>{for(const id of ["source","startAct","goalAct","component","session","outcome","startFloorMin","startFloorMax","horizon","snapshotRole"])$(id).value="";saveChoices();state.scatterDomain=null;scheduleEpisodes();});
for(const id of ["bins","includeCuts","floorHistMetric","healthMetric"])$(id).addEventListener("change",()=>{saveChoices();state.scatterDomain=null;renderEpisodes();renderTraining();});
for(const id of ["scatterX","scatterY","scatterGroup"])$(id).addEventListener("change",()=>{saveChoices();state.scatterDomain=null;renderScatter();});
$("scatterReset").addEventListener("click",()=>{state.scatterDomain=null;renderScatter();});$("downloadCSV").addEventListener("click",downloadCSV);
let pointerFrame=null;
window.addEventListener("pointermove",event=>{
  const drag=state.drag;if(!drag)return;
  if(pointerFrame)cancelAnimationFrame(pointerFrame);
  pointerFrame=requestAnimationFrame(()=>{
    if(state.drag!==drag)return;
    if(drag.kind==="range") {const shift=(event.clientX-drag.startX)/drag.pixels*(drag.range[1]-drag.range[0]);setRange([drag.range[0]-shift,drag.range[1]-shift],false);}
    else if(drag.kind==="brush"){const [low,high]=drag.bounds,round=low+(event.clientX-drag.rect.left)/drag.rect.width*(high-low);setRange([Math.min(drag.startRound,round),Math.max(drag.startRound,round)],false);}
    else if(drag.kind==="scatter") {const dx=(event.clientX-drag.startX)/drag.width*(drag.domain.x[1]-drag.domain.x[0]),dy=(event.clientY-drag.startY)/drag.height*(drag.domain.y[1]-drag.domain.y[0]);state.scatterDomain={x:[drag.domain.x[0]-dx,drag.domain.x[1]-dx],y:[drag.domain.y[0]+dy,drag.domain.y[1]+dy]};renderScatter();}
  });
});
window.addEventListener("pointerup",()=>{if(pointerFrame){cancelAnimationFrame(pointerFrame);pointerFrame=null;}if(state.drag?.kind!=="scatter"&&state.drag)scheduleEpisodes();state.drag=null;});
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state.data){renderTraining();renderEpisodes();renderFullRuns();renderAudits();}},150);});
setInterval(()=>{if($("auto").checked&&!document.hidden)refresh();},8000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden&&$("auto").checked)refresh();});
refresh();
}
