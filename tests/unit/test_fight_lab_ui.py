"""Exercise UI event handlers with a small DOM fake; no browser dependency."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "requires node for JavaScript unit test")
class FightLabUiTest(unittest.TestCase):
    def test_all_phases_and_run_uses_the_displayed_source_state_once(self):
        script = Path(__file__).resolve().parents[2] / "src/sts_ai/interactive/static/fight_lab.js"
        code = r'''
const vm = require('node:vm'), fs = require('node:fs'), assert = require('node:assert/strict');
const nodes = new Map(), calls = [];
function element(id) {
  if (!nodes.has(id)) nodes.set(id, {value:'sources/seed_3_r0.jsonl', selectedOptions:[{textContent:'Source'}],
    setAttribute(k,v){this[k]=v}, removeAttribute(k){delete this[k]}});
  return nodes.get(id);
}
const row = {state_id:'seed_3_r0_d9',phase:'combat',enemies:['CULTIST'],world_seed:3,floor:1,turn:2,index:9};
const context = vm.createContext({document:{getElementById:element}, URLSearchParams,
  FormData: class { *[Symbol.iterator]() {yield ['phase',''];yield ['enemy',''];} },
  clearInterval(){}, setTimeout(){},
  fetch: async (url,options) => {
    if (url === '/api/collection') return new Promise(()=>{}); // Don't start page initialization.
    calls.push({url,options});
    if (url.startsWith('/api/view?')) return {ok:true,json:async()=>({index:9,count:20,
      source_state:row,board_html:'board',record:{},outcome:'max_decisions'})};
    if (url === '/api/runs' && options?.method === 'POST') return {ok:true,json:async()=>({
      job_id:'job',state_id:JSON.parse(options.body).state_id,status:'running',decisions:0})};
    throw Error('unexpected request '+url);
  }});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
assert.equal(vm.runInContext('params().has("phase")',context),true);
assert.equal(vm.runInContext('params().get("phase")',context),'');
(async()=>{
  context.row = row;
  vm.runInContext('current = row; runTarget = {...row,state_id:"wrong_old_state"}',context);
  await vm.runInContext('view(9)',context);
  assert.equal(element('runModel').disabled,false);
  await element('runModel').onclick();
  assert.equal(JSON.parse(calls.find(c=>c.options?.method==='POST').options.body).state_id,row.state_id);
  assert.equal(element('runModel').disabled,true);
  assert.equal(element('cancelRun').hidden,false);
  assert.match(element('runStatus').textContent,/Running seed_3_r0_d9/);
  await element('runModel').onclick();
  assert.equal(calls.filter(c=>c.options?.method==='POST').length,1);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
        result = subprocess.run([shutil.which("node"), "-e", code, str(script)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
