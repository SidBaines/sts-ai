function indexForTurn(frames, target) {
  if (target === 'end') return frames.length;
  const found = frames.findIndex(f => f.turn === Number(target));
  return found < 0 ? frames.length : found;
}
if (typeof module !== 'undefined') module.exports = {indexForTurn};

if (typeof document !== 'undefined') (() => {
  const data = JSON.parse(document.getElementById('data').textContent);
  const ids = ['old_format', 'new_no_thinking', 'new_thinking'];
  const colors = ['#aec0d8', '#75d6bb', '#c6a5f0'];
  const labels = ['Old format · no thinking', 'New format · no thinking', 'New format · thinking'];
  const panels = {};
  const $ = id => document.getElementById(id);
  const turnName = t => t === 0 ? 'Opening' : `Turn ${t}`;
  const hp = state => state?.combat?.player_cur_hp ?? state?.cur_hp ?? '—';
  const duration = seconds => seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(1)} sec`;
  function node(tag, text, className) {
    const n = document.createElement(tag); if (text !== undefined) n.textContent = text;
    if (className) n.className = className; return n;
  }
  function option(select, value, label) {
    const o = node('option', label); o.value = value; select.append(o);
  }
  function setIndex(id, index, notice = '') {
    const p = panels[id], arm = data.arms[id];
    p.index = Math.max(0, Math.min(index, arm.records.length)); p.notice = notice;
    render(id);
  }
  function render(id) {
    const p = panels[id], a = data.arms[id], n = a.records.length, end = p.index === n;
    p.slider.value = p.index; p.select.value = p.index;
    p.prev.disabled = p.index === 0; p.next.disabled = end;
    const r = end ? null : a.records[p.index];
    p.location.textContent = end ? 'Final outcome · after the last action' : `${turnName(a.frames[p.index].turn)} · Decision ${p.index + 1} of ${n} · HP ${hp(r.state)} → ${hp(r.after_state)}`;
    p.action.textContent = end ? `${a.result.metrics.label} · ${a.result.metrics.exit_hp} HP remaining` : r.selected_action?.description || '(no executed action)';
    p.tabs.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.tab === p.tab));
    const content = p.content; content.replaceChildren();
    if (p.notice) content.append(node('p', p.notice, 'notice'));
    if (p.tab === 'actions') {
      const list = node('div', undefined, 'action-list');
      a.records.forEach((item, i) => {
        const button = node('button', `${i + 1} · ${turnName(a.frames[i].turn)} · HP ${hp(item.state)} → ${hp(item.after_state)} · ${item.selected_action?.description || '(none)'}`);
        button.classList.toggle('active', i === p.index);
        button.onclick = () => {p.tab = 'board'; setIndex(id, i);}; list.append(button);
      });
      content.append(list); return;
    }
    if (end) {
      const final = node('div', undefined, 'final'); final.append(node('p', 'Fight complete'), node('strong', `${a.result.metrics.exit_hp} HP`), node('p', `${a.result.metrics.label} · ${n} decisions · ${duration(a.result.seconds)}`));
      content.append(final, node('p', 'Use Previous to inspect the last decision, or Actions to jump to any earlier choice.'));
      const detail = node('details'); detail.append(node('summary', 'Recorded final state'), node('pre', JSON.stringify(a.records.at(-1).after_state, null, 2))); content.append(detail); return;
    }
    if (p.tab === 'board') {
      const board = node('div'); board.innerHTML = a.frames[p.index].board_html;
      board.querySelectorAll('.sts-enemy .sts-enm').forEach((label, i) => {
        const enemy = r.state.combat?.enemies?.[i];
        if (enemy) label.textContent += ` · enemy ${enemy.index}`;
      });
      content.append(board);
    } else if (p.tab === 'observation') {
      content.append(node('h3', 'Saved observation'), node('pre', r.state_text), node('h3', 'Legal actions'), node('pre', JSON.stringify(r.legal_actions, null, 2)));
    } else if (p.tab === 'thinking') {
      content.append(node('p', `${r.agent.thinking_tokens || 0} thinking tokens · ${r.agent.completion_tokens} total output tokens · ${duration(r.agent.latency_s)}`),
        node('pre', r.agent.thinking || 'No separate thinking text was recorded. The Response tab includes any prose emitted with the action.'));
    } else if (p.tab === 'response') {
      content.append(node('p', `Recorded action: ${r.agent.valid ? 'valid' : 'invalid'} · ${r.agent.completion_tokens} tokens · ${duration(r.agent.latency_s)}`), node('pre', r.agent.raw_response));
    } else content.append(node('pre', JSON.stringify(r, null, 2)));
  }
  ids.forEach((id, j) => {
    const a = data.arms[id], column = node('section', undefined, 'column'); column.id = id; column.style.setProperty('--arm', colors[j]);
    const head = node('div', undefined, 'panel-head'); head.append(node('h2', labels[j]), node('div', `${a.result.metrics.label} · ${a.result.metrics.exit_hp} HP left · started at ${a.result.metrics.entry_hp}`, 'outcome'), node('div', `${a.records.length} decisions · ${duration(a.result.seconds)} · ${a.settings.max_tokens.toLocaleString()} token limit`, 'small'));
    const controls = node('div', undefined, 'controls'); const prev = node('button', '←'), next = node('button', '→'), select = node('select');
    prev.setAttribute('aria-label', `${labels[j]} previous decision`); next.setAttribute('aria-label', `${labels[j]} next decision`); select.setAttribute('aria-label', `${labels[j]} decision`);
    a.frames.forEach((f, i) => option(select, i, `${i + 1} · ${turnName(f.turn)} · ${a.records[i].selected_action?.description || 'No action'}`)); option(select, a.records.length, 'Final outcome');
    controls.append(prev, select, next); const slider = node('input', undefined, 'slider'); slider.type = 'range'; slider.min = 0; slider.max = a.records.length; slider.setAttribute('aria-label', `${labels[j]} timeline`);
    const location = node('div', undefined, 'location'); head.append(controls, slider, location);
    const action = node('div', undefined, 'action'), tabs = node('div', undefined, 'tabs'), content = node('div', undefined, 'content');
    for (const [tab, label] of [['board','Board'],['observation','Observation'],['thinking','Thinking'],['response','Response'],['actions','Actions'],['record','JSON']]) {
      const b = node('button', label); b.dataset.tab = tab; b.onclick = () => {panels[id].tab = tab; render(id);}; tabs.append(b);
    }
    panels[id] = {index:0,tab:'board',column,prev,next,select,slider,location,action,tabs,content};
    prev.onclick = () => setIndex(id, panels[id].index - 1); next.onclick = () => setIndex(id, panels[id].index + 1);
    select.onchange = () => setIndex(id, Number(select.value)); slider.oninput = () => setIndex(id, Number(slider.value));
    column.append(head, action, tabs, content); $('columns').append(column); option($('focus'), id, labels[j]); render(id);
  });
  const turns = [...new Set(ids.flatMap(id => data.arms[id].frames.map(f => f.turn)))].filter(t => t !== null).sort((a,b) => a-b);
  turns.forEach(t => option($('turn'), t, turnName(t))); option($('turn'), 'end', 'Final outcomes');
  function jump(target) {
    $('turn').value = target;
    ids.forEach(id => {
      const frames = data.arms[id].frames, index = indexForTurn(frames, target);
      setIndex(id, index, target !== 'end' && index === frames.length ? `This trajectory has no ${turnName(Number(target)).toLowerCase()}; showing its final outcome.` : '');
    });
    $('prevTurn').disabled = $('turn').selectedIndex === 0;
    $('nextTurn').disabled = $('turn').selectedIndex === $('turn').options.length - 1;
  }
  $('turn').onchange = () => jump($('turn').value);
  $('start').onclick = () => jump(turns[0]); $('end').onclick = () => jump('end');
  $('prevTurn').onclick = () => jump($('turn').options[Math.max(0,$('turn').selectedIndex-1)].value);
  $('nextTurn').onclick = () => jump($('turn').options[Math.min($('turn').options.length-1,$('turn').selectedIndex+1)].value);
  $('focus').onchange = () => {
    const chosen = $('focus').value; $('columns').classList.toggle('focused', chosen !== 'all');
    ids.forEach(id => panels[id].column.classList.toggle('hidden', chosen !== 'all' && chosen !== id));
  };
  $('title').textContent = `${data.title} · Floor ${data.floor}`;
  document.title = `${data.title} · Three trajectories`;
  $('subtitle').textContent = `World ${data.world_seed} · Same saved starting position and sampling seed · ${data.state_id}`;
  jump(turns[0]);
})();
