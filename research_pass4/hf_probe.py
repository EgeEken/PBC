import ast
import datetime
import json
import time
import urllib.error
import urllib.request

from support import ROOT, OUT, save, sha, snapshot


def main():
    before = snapshot()
    source = (ROOT / 'server.py').read_text(encoding='utf-8')
    routes = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and isinstance(d.func.value, ast.Name) and d.func.value.id == 'app' and d.args and isinstance(d.args[0], ast.Constant):
                routes.append(dict(method=d.func.attr.upper(), path=d.args[0].value,
                    handler=node.name, line=node.lineno))
    save('local_endpoint_inventory', routes)
    base = 'https://egeeken-pbc.hf.space'
    urls = [base + p for p in ('/api/health', '/openapi.json', '/api/benchmark/status',
        '/api/quick_rd/status', '/api/sweeps/run/status')]
    urls.append('https://huggingface.co/api/spaces/EgeEken/PBC')
    rows = []
    for url in urls:
        started = time.perf_counter()
        row = dict(url=url, method='GET', time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PBC-Pass4-readonly-audit'})
            with urllib.request.urlopen(req, timeout=25) as response:
                payload = response.read(2_000_001)
                assert len(payload) <= 2_000_000
                row.update(status=response.status, headers={k:v for k,v in response.headers.items()
                    if k.lower() in ('content-type', 'date', 'server', 'x-repo-commit', 'x-revision')},
                    body=json.loads(payload), body_sha256=sha(payload))
        except Exception as error:
            row.update(error=type(error).__name__, message=str(error))
        row['http_seconds'] = time.perf_counter()-started
        rows.append(row)
        print(url, row.get('status', row.get('error')), flush=True)
        save('hf_observations', rows)
    schema = next((r.get('body') for r in rows if r['url'].endswith('/openapi.json') and r.get('status') == 200), None)
    if schema:
        save('remote_endpoint_inventory', [dict(path=p, method=m.upper(), operation=v.get('operationId'),
            security=v.get('security', schema.get('security')), request=v.get('requestBody'), parameters=v.get('parameters'))
            for p, methods in schema.get('paths', {}).items() for m,v in methods.items()])
    metadata = next((r.get('body') for r in rows if r['url'].startswith('https://huggingface.co/api/') and r.get('status') == 200), {})
    revision = metadata.get('runtime', {}).get('sha')
    checks = []
    if revision:
        for name in ('server.py', 'pbc3_benchmark.py', 'pbc3_quick_rd.py', 'PBC3.py',
                     'pbc3_ops.py', 'pbc3_stream.py', 'learned_filler.py', 'Dockerfile', 'requirements_docker.txt'):
            url = f'https://huggingface.co/spaces/EgeEken/PBC/resolve/{revision}/{name}'
            try:
                with urllib.request.urlopen(url, timeout=25) as response:
                    payload = response.read(1_000_001)
                    assert len(payload) <= 1_000_000
                text = payload.decode('utf-8').replace('\r\n', '\n')
                local = (ROOT / name).read_text(encoding='utf-8').replace('\r\n', '\n')
                checks.append(dict(path=name, url=url, revision=revision,
                    remote_sha256_normalized=sha(text.encode()), local_sha256_normalized=sha(local.encode()), equal=text == local))
                (OUT / ('deployed_' + name.replace('.', '_') + '.txt')).write_text(text, encoding='utf-8')
            except Exception as error:
                checks.append(dict(path=name, error=str(error)))
    save('hf_reported_revision_source_checks', checks)
    save('hf_probe_provenance', dict(before=before, after=snapshot(), post_requests=0, uploads=0,
        codec_benchmarks=0, reason='Existing start API has no bounded image/preset selection or experimental variants; minimum PBC run is 20 images x four presets and appends DB rows. Quick RD overwrites shared results.'))


if __name__ == '__main__':
    main()
