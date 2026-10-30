import json
from pathlib import Path
import sys
import time
import tkinter as tk
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pbc_gui as gui
OUT=HERE/'artifacts/gui-tests'
OUT.mkdir(exist_ok=True)
root=tk.Tk()
root.withdraw()
app=gui.App(root)
errors=[]
gui.messagebox.showerror=lambda title,text,**kw:errors.append(str(text))
assert [app.tabs.tab(tab,'text') for tab in app.tabs.tabs()]==['Encode','Decode','Registry']
assert app.preset.get()=='quality' and app.mode.get()=='Auto'
def wait():
    start=time.perf_counter()
    while app.busy or app.pending_encode or (app.warmup_future and not app.warmup_done):
        root.update()
        time.sleep(.01)
        assert time.perf_counter()-start<180,'GUI job timeout'
    root.update()
    return time.perf_counter()-start
reference=HERE/'artifacts/probe-candidate'
fixtures=json.loads((HERE.parent/'research_pass28/smoke/fixtures.json').read_text())
checks=[]
unicode_dir=OUT/'Unicode Çığ 日本 with spaces'
unicode_dir.mkdir(exist_ok=True)
for fixture in fixtures['inputs']:
    name=fixture['name']
    path=unicode_dir/(name+'.png')
    path.write_bytes((reference/(name+'.png')).read_bytes())
    for preset in gui.PRESETS:
        before=app.heartbeats
        app.preset.set(preset)
        app.path_vars['encode'].set(str(path))
        app.start_encode()
        assert app.busy or app.pending_encode
        duration=wait()
        assert not errors,errors
        entry=app.results['encode']
        assert Path(entry['pbc']).read_bytes()==(reference/(name+'.'+preset+'.pbc')).read_bytes()
        assert 'Native C' in entry['stats']
        assert app.heartbeats>before
        target=unicode_dir/(name+'.'+preset+'.pbc')
        gui.export(entry,target,'pbc')
        assert target.read_bytes()==Path(entry['pbc']).read_bytes()
        app.path_vars['decode'].set(str(target))
        app.start_decode()
        wait()
        assert not errors,errors
        png=unicode_dir/(name+'.'+preset+'.decoded.png')
        gui.export(app.results['decode'],png,'png')
        with gui.Image.open(png) as actual,gui.Image.open(reference/(name+'.'+preset+'.png')) as expected:
            assert actual.mode==expected.mode and actual.size==expected.size and actual.tobytes()==expected.tobytes()
        checks.append(dict(name=name,preset=preset,heartbeats=app.heartbeats-before,wall_seconds=duration))
app.mode.set('Manual')
app.preset.set('quality')
app.refresh_parameters()
for name,value in dict(patch_count=7,search_depth=15,proposal_depth=8,exact_depth=3,use_lzma=False,color_space='RGB',downsample_rate=1).items():
    app.parameter_vars[name].set(str(value))
app.start_encode()
wait()
assert not errors,errors
assert app.results['encode']['config']['patch_count']==7
assert app.results['encode']['config']['compute_final_mse']
app.parameter_vars['min_patch_size'].set('900')
app.parameter_vars['max_patch_size'].set('8')
app.start_encode()
assert not app.busy and errors
errors.clear()
for value in ('nan','inf','-1','1.1'):
    config=gui.PBC3Config.quality()
    values={f.name:str(getattr(config,f.name)) for f in gui.dataclasses.fields(config)}
    values['learned_filler_q']=value
    try:gui.configuration('quality','Manual',values)
    except ValueError:pass
    else:raise AssertionError('Invalid quality accepted '+value)
app.mode.set('Semi-Auto')
app.refresh_parameters()
assert set(app.parameter_vars)==gui.SEMI
app.parameter_vars['patch_count'].set('9')
assert gui.configuration('quality','Semi-Auto',{k:v.get() for k,v in app.parameter_vars.items()}).patch_count==9
app.start_encode()
wait()
assert app.results['encode']['config']['patch_count']==9
assert not errors,errors
bad=OUT/'corrupt.pbc'
bad.write_bytes(b'PBC3\0\0bad')
app.path_vars['decode'].set(str(bad))
app.start_decode()
wait()
assert errors
errors.clear()
for variant in ('original','published'):
    stream=HERE/'artifacts/streams'/variant/'eiffel.png.quality.pbc'
    app.path_vars['decode'].set(str(stream))
    app.start_decode()
    wait()
    assert not errors,errors
entry=app.results['decode']
app.select_registry(entry['id'])
app.select_registry()
app.reopen()
assert app.tabs.select()==str(app.decode_tab)
seen={}
saved=OUT/'overwrite.png'
saved.write_bytes(b'previous')
def choose(**kwargs):
    seen.update(kwargs)
    return str(saved)
gui.filedialog.asksaveasfilename=choose
app.save('decode','png')
assert seen['confirmoverwrite'] is True
with gui.Image.open(saved) as image:assert image.width>0
try:gui.export(dict(entry,source=str(saved)),saved,'png')
except ValueError:pass
else:raise AssertionError('Source overwrite accepted')
chosen=unicode_dir/'RGB.png'
gui.filedialog.askopenfilename=lambda **kw:str(chosen)
app.browse('encode')
wait()
assert app.input_filename.get()=='RGB.png'
assert app.input_preview.picture is not None
assert 'encode' not in app.results
assert app.encode_save.cget('state')=='disabled'
app.mode.set('Manual')
app.refresh_parameters()
assert set(app.parameter_vars)=={f.name for f in gui.dataclasses.fields(gui.PBC3Config())}-gui.HIDDEN
app.editor.update_idletasks()
last=next(w for w in app.editor.winfo_children() if isinstance(w,gui.ttk.Entry)
    and w.cget('textvariable')==str(app.parameter_vars['random_seed']))
last.event_generate('<FocusIn>')
root.update()
assert app.settings_canvas.yview()[0]>0
assert app.compress_button.cget('takefocus')=='1'

tiny=gui.Image.new('RGB',(11,7))
for i in range(31):
    result=app.stage(dict(image=tiny,data=b'fixture',seconds=.01,bpp=1))
    app.add_result('decode','fixture.pbc',result)
assert len(app.entries)==30 and len(app.registry_cards)==30
assert len(list(Path(app.storage.name).iterdir()))==60
root.update()
result=dict(encode_decode_cases=checks,manual_override=True,invalid_parameters=True,
    semi_auto=True,registry_limit=30,registry_reopen=True,invalid_stream_error=True,
    unicode_paths=True,exports=True,overwrite_dialog_requested=True,source_overwrite_rejected=True,
    historical_decode=True,ui_heartbeat_during_jobs=True,default_preset='quality')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
app.close()
print(json.dumps(result,indent=2))
