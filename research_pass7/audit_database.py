import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / 'database_before.db'
    con = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute('SELECT * FROM results ORDER BY id')]
    con.close()
    runs = []
    for run_id in dict.fromkeys(r['run_id'] for r in rows):
        group = [r for r in rows if r['run_id'] == run_id]
        pbc = [r for r in group if r['codec'] == 'PBC3']
        runs.append(dict(run_id=run_id, first_utc=datetime.datetime.fromtimestamp(min(r['ts'] for r in group), datetime.timezone.utc).isoformat(),
            last_utc=datetime.datetime.fromtimestamp(max(r['ts'] for r in group), datetime.timezone.utc).isoformat(),
            rows=len(group), codecs=sorted({r['codec'] for r in group}),
            pbc_rows=len(pbc), images=len({r['image'] for r in pbc}), trials=sorted({r['trial'] for r in pbc}),
            presets=sorted({r['variant'] for r in pbc}),
            encode_seconds=mean(r['encode_seconds'] for r in pbc) if pbc else None,
            decode_seconds=mean(r['decode_seconds'] for r in pbc) if pbc else None))
    pbc = [r for r in rows if r['codec'] == 'PBC3']
    result = dict(database_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(rows), runs=runs,
        pbc_rows=len(pbc), all_history={k:mean(r[k] for r in pbc) for k in ('encode_seconds','decode_seconds')},
        trial_means=[dict(run_id=g['run_id'],trial=t,**{k:mean(r[k] for r in pbc if r['run_id']==g['run_id'] and r['trial']==t) for k in ('encode_seconds','decode_seconds')}) for g in runs for t in g['trials']])
    repository=HERE/'repository_database.db'
    if repository.exists():
        con=sqlite3.connect(f'{repository.as_uri()}?mode=ro',uri=True)
        con.row_factory=sqlite3.Row
        result['repository_runs']=[dict(r) for r in con.execute('SELECT run_id, codec, COUNT(*) count, AVG(encode_seconds) encode_seconds, AVG(decode_seconds) decode_seconds FROM results GROUP BY run_id,codec')]
        result['repository_database_sha256']=hashlib.sha256(repository.read_bytes()).hexdigest()
        con.close()
    (HERE/'database_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
