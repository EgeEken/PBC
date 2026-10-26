const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, { value: '', options: [], replacements: 0,
      replaceChildren() { this.options = []; this.replacements++; }, add(option) { this.options.push(option); } });
    return nodes.get(id);
  };
  const pending = [];
  const context = { window: {}, document: {getElementById: node}, URLSearchParams, Option: function(text, value) {this.text=text; this.value=value;},
    fetch: url => new Promise(resolve => pending.push({url,resolve: data => resolve({json: async () => data})})) };
  const source = fs.readFileSync(path.join(__dirname,'../static/benchmark.js'),'utf8')
    .replace('return { onShow, onHide, refreshResults };','return { refreshResults, S, applyFilters, avgByCodec };');
  vm.runInNewContext(source,context);
  const page=context.window.BenchmarkPage;
  page.S.built=true;
  const runs=[{run_id:'one',first_ts:1,status:'complete coverage (inferred)',codecs:['PBC3'],trials:1,images:2,count:8}];
  const point={codec:'PBC3',variant:'quality',count:8,psnr:30,mse:10,bpp:.5,encode_seconds:1,decode_seconds:.1};
  node('bm-run-select').value='latest';
  const old=page.refreshResults();
  node('bm-run-select').value='all';
  const current=page.refreshResults();
  pending[1].resolve({runs,points:[point],count:8,selection_status:'historical composite',selected_run:null});
  await current;
  pending[0].resolve({runs,points:[{...point,encode_seconds:99}],count:8,selected_run:'one'});
  await old;
  assert.equal(node('bm-run-select').value,'all');
  assert.equal(page.avgByCodec('encode_seconds')[0].v,1);
  assert.match(node('bm-run-info').textContent,/historical composite/);
  const replacements=node('bm-run-select').replacements;
  const poll=page.refreshResults();
  pending[2].resolve({runs,points:[point],count:8,selection_status:'historical composite',selected_run:null});
  await poll;
  assert.equal(node('bm-run-select').replacements,replacements);
  node('bm-encmax').value='.5'; page.applyFilters();
  assert.equal(page.S.filtered.length,0);
  node('bm-encmax').value=''; page.applyFilters();
  assert.equal(page.S.filtered.length,1);
  const output={passed:true,stale_response_ignored:true,stable_options_not_replaced:true,chart_filter_preserved:true};
  fs.writeFileSync(path.join(__dirname,'selector_race_checks.json'),JSON.stringify(output,null,2)+'\n');
  console.log(JSON.stringify(output));
}
main().catch(error => { console.error(error); process.exitCode=1; });
