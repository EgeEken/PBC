import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import pbc_gui as gui
from pbc3_decoder import decode,read_file,MAX_FILE
from pbc3_native import DLL,NativeMSE,select_metric
assert getattr(sys,'frozen',False)
assert os.environ['PATH'].lower()==r'c:\windows\system32'
if '--latency' in sys.argv:
    sys.argv=[sys.argv[0],'after','--root',sys.argv[2]]
    import profile_ui
    sys.exit(0)
if '--features' in sys.argv:
    import test_final_gui
    sys.exit(0)
if '--warmup' in sys.argv:
    from pbc_startup import signatures
    output=Path(sys.argv[2])
    source=Path(sys.argv[3])
    root=tk.Tk()
    root.withdraw()
    app=gui.App(root)
    errors=[]
    gui.messagebox.showerror=lambda title,text,**kw:errors.append(str(text))
    root.deiconify()
    started=time.perf_counter()
    gaps=[]
    last=started
    while not app.warmup_done:
        root.update()
        now=time.perf_counter()
        gaps.append(now-last)
        last=now
        time.sleep(.005)
        assert now-started<180
    assert not app.entries and app.warmup_info['native']
    startup=dict(window_visible_seconds=app.visible_seconds,warmup=app.warmup_info,
        initialization_wall_seconds=time.perf_counter()-started,poll_ticks=app.heartbeats,
        event_loop_gap_max_seconds=max(gaps),registry_entries_before_encode=len(app.entries))
    app.loaded(str(source),gui.load_image(source))
    runs=[]
    for index in range(2):
        app.start_encode()
        while app.busy or app.pending_encode:
            root.update()
            time.sleep(.005)
        assert not errors,errors
        entry=app.results['encode']
        runs.append(dict(seconds=entry['seconds'],jit_added=app.last_jit_added,
            stream_sha256=hashlib.sha256(Path(entry['pbc']).read_bytes()).hexdigest()))
    startup['encodes']=runs
    startup['cache_files']=len(list(Path(os.environ['NUMBA_CACHE_DIR']).rglob('*.nbc')))
    app.close()
    output.write_text(json.dumps(startup,indent=2)+'\n')
    print(json.dumps(startup,indent=2))
    sys.exit(0)
if '--worker' in sys.argv:
    import ctypes as ct
    from PIL import ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    assert ct.windll.kernel32.SetProcessAffinityMask(ct.c_void_p(-1),ct.c_size_t(1<<10))
    images={}
    for path in sys.argv[2:]:
        with gui.Image.open(path) as image:
            images[Path(path).name]=ImageOps.exif_transpose(image).convert('RGB')
    assert isinstance(select_metric('auto',lambda a,b:None),NativeMSE)
    print(json.dumps(dict(ready=True,native_dll=str(DLL),affinity=[10])),flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        image=images[Path(req['image']).name]
        config=getattr(PBC3Config,req['preset'])()
        started=time.perf_counter()
        result=PBC3.compress(image,config=config)
        enc=time.perf_counter()-started
        started=time.perf_counter()
        decoded=PBC3.decompress(result.data).image
        dec=time.perf_counter()-started
        print(json.dumps(dict(encode_seconds=enc,decode_seconds=dec,bytes=len(result.data),
            bpp=8*len(result.data)/(image.width*image.height),internal_mse=result.mse,
            stream_sha256=hashlib.sha256(result.data).hexdigest(),
            pixel_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),error=None)),flush=True)
    sys.exit(0)
fixture=json.loads(Path(sys.argv[1]).read_text())
out=Path(sys.argv[2])
out.parent.mkdir(parents=True,exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
root=tk.Tk()
root.withdraw()
app=gui.App(root)
errors=[]
gui.messagebox.showerror=lambda title,text,**kw:errors.append(str(text))
def wait():
    started=time.perf_counter()
    while app.busy or app.pending_encode or (app.warmup_future and not app.warmup_done):
        root.update()
        time.sleep(.01)
        if time.perf_counter()-started>180:raise TimeoutError('GUI job timeout')
    root.update()
    assert not errors,errors
    return time.perf_counter()-started
checks=[]
with tempfile.TemporaryDirectory(prefix='PBC Frozen Çığ 日本 ') as directory:
    folder=Path(directory)
    for row in fixture['inputs']:
        path=folder/(row['name']+' input.png')
        path.write_bytes(base64.b64decode(row['png']))
        for preset in gui.PRESETS:
            app.preset.set(preset)
            app.path_vars['encode'].set(str(path))
            before=app.heartbeats
            app.start_encode()
            duration=wait()
            entry=app.results['encode']
            data=Path(entry['pbc']).read_bytes()
            expected=next(r for r in fixture['expected'] if r['name']==row['name'] and r.get('preset')==preset)
            assert sha(data)==expected['stream']
            image=decode(data)
            assert sha(image.tobytes())==expected['pixels']
            assert 'Native C' in entry['stats']
            saved=folder/(row['name']+'.'+preset+'.pbc')
            gui.export(entry,saved,'pbc')
            app.path_vars['decode'].set(str(saved))
            app.start_decode()
            wait()
            target=folder/'reconstruction.png'
            gui.export(app.results['decode'],target,'png')
            with gui.Image.open(target) as png:
                assert sha(png.tobytes())==expected['pixels']
            checks.append(dict(name=row['name'],preset=preset,stream=sha(data),
                pixels=sha(image.tobytes()),wall_seconds=duration,heartbeats=app.heartbeats-before,
                encode_seconds=entry['seconds']))
    app.mode.set('Manual')
    app.refresh_parameters()
    app.parameter_vars['patch_count'].set('7')
    app.start_encode()
    wait()
    assert app.results['encode']['config']['patch_count']==7
    assert app.results['encode']['config']['compute_final_mse']
    known=0
    partials=0
    for row in fixture['valid']:
        data=base64.b64decode(row['data'])
        image=decode(data)
        assert sha(image.tobytes())==row['pixel_sha256']
        assert list(image.size)==row['size'] and image.mode==row['mode']
        for limit,expected in row['partial'].items():
            assert sha(decode(data,int(limit)).tobytes())==expected
            partials+=1
        known+=1
    rejected=0
    for row in fixture['invalid']:
        try:decode(base64.b64decode(row))
        except (ValueError,EOFError,IndexError,KeyError,OverflowError):rejected+=1
        else:raise AssertionError('Invalid stream accepted')
    oversized=folder/'oversized.pbc'
    with oversized.open('wb') as f:f.truncate(MAX_FILE+1)
    try:read_file(oversized)
    except ValueError:rejected+=1
    else:raise AssertionError('Oversized file accepted')
assert isinstance(select_metric('auto',lambda a,b:None),NativeMSE)
assert Path(DLL).is_relative_to(Path(sys._MEIPASS))
result=dict(checks=checks,manual_override=True,known_streams=known,partial_checks=partials,
    invalid_rejected=rejected,warmup=app.warmup_info,native_dll=str(DLL),native_dll_sha256=sha(Path(DLL).read_bytes()),
    cache=os.environ['NUMBA_CACHE_DIR'],cached_files=len(list(Path(os.environ['NUMBA_CACHE_DIR']).rglob('*.nbc'))),
    path=os.environ['PATH'],python=sys.version,module_origin=str(sys.modules['PBC3'].__file__),
    registry_entries=len(app.entries),frozen=True)
app.close()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
