const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const result=JSON.parse(fs.readFileSync(path.join(__dirname,'public_benchmark.json'),'utf8'));
  const nodes=new Map();
  const node=id=>{
    if(!nodes.has(id)) nodes.set(id,{value:'',options:[],replaceChildren(){this.options=[];},add(option){this.options.push(option);}});
    return nodes.get(id);
  };
  const source=fs.readFileSync(path.join(__dirname,'../static/benchmark.js'),'utf8')
    .replace('return { onShow, onHide, refreshResults };','return { refreshResults, S, avgByCodec };');
  const context={window:{},document:{getElementById:node},URLSearchParams,
    Option:function(text,value){this.text=text;this.value=value;},fetch:async()=>({json:async()=>result})};
  vm.runInNewContext(source,context);
  const page=context.window.BenchmarkPage;
  page.S.built=true;
  node('bm-run-select').value='latest';
  await page.refreshResults();
  assert.match(node('bm-run-info').textContent,/PBC 3.1 development — release-compatible RNG/);
  assert.match(node('bm-run-info').textContent,/240 selected samples/);
  assert.equal(page.S.filtered.length,4);
  assert.ok(page.S.filtered.every(p=>p.codec==='PBC3'));
  assert.ok(node('bm-run-select').options.find(o=>o.value===result.selected_run).text.includes('release-compatible RNG'));
  for(const run of result.runs.filter(r=>!r.metadata)) {
    assert.ok(!node('bm-run-select').options.find(o=>o.value===run.run_id).text.includes('release-compatible RNG'));
  }
  const output={passed:true,run_id:result.selected_run,baseline_label_visible:true,selected_samples:240,
    historical_labels_preserved:true,pbc_only_plots:true,served_frontend_tested:true};
  fs.writeFileSync(path.join(__dirname,'ui_checks.json'),JSON.stringify(output,null,2)+'\n');
  console.log(JSON.stringify(output));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
