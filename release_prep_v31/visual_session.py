import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
ct.windll.user32.SetProcessDpiAwarenessContext(ct.c_void_p(-4))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pbc_gui as gui
p=argparse.ArgumentParser()
p.add_argument('state',choices=('empty','selected','encoded','decoded','registry','manual'))
p.add_argument('--size',choices=('normal','minimum','maximized'),default='normal')
p.add_argument('--scale',type=int,choices=(100,125,150))
a=p.parse_args()
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts/ui-polish/screenshots'
OUT.mkdir(parents=True,exist_ok=True)
root=gui.tk.Tk()
root.withdraw()
if a.scale:
    root.tk.call('tk','scaling',96*a.scale/100/72)
app=gui.App(root)
source=HERE.parent/'hpt_data/cat.jpg'
app.root.title('PBC V3.1 — Review '+a.state+' '+a.size+(' '+str(a.scale) if a.scale else ' native'))
errors=[]
gui.messagebox.showerror=lambda title,text,**kw:errors.append(str(text))
def wait():
    start=time.perf_counter()
    while app.busy or app.pending_encode or (app.warmup_future and not app.warmup_done):
        root.update()
        time.sleep(.01)
        assert time.perf_counter()-start<180
    assert not errors,errors
    root.update()
if a.state!='empty':
    app.path_vars['encode'].set(str(source))
    app.loaded(str(source),gui.load_image(source))
if a.state in ('encoded','decoded','registry','manual'):
    app.start_encode()
    wait()
if a.state in ('decoded','registry'):
    target=OUT/'review-cat.pbc'
    gui.export(app.results['encode'],target,'pbc')
    app.path_vars['decode'].set(str(target))
    app.start_decode()
    wait()
if a.state=='registry':
    app.path_vars['encode'].set(str(HERE.parent/'hpt_data/flower.jpg'))
    app.start_encode()
    wait()
    app.tabs.select(app.registry_tab)
elif a.state=='decoded':
    app.tabs.select(app.decode_tab)
elif a.state=='manual':
    app.mode.set('Manual')
root.deiconify()
root.update()
if a.size=='minimum':
    root.geometry(f'{gui.px(820)}x{gui.px(650)}')
elif a.size=='maximized':
    root.state('zoomed')
root.update()
name=a.state+'-'+a.size+'-'+(str(a.scale) if a.scale else 'native')
metadata=dict(state=a.state,size=a.size,requested_render_scale=a.scale,
    tk_dpi=root.winfo_fpixels('1i'),os_dpi=ct.windll.user32.GetDpiForWindow(root.winfo_id()),
    window=[root.winfo_width(),root.winfo_height()],source=str(source),
    entries=[dict(operation=e['operation'],source=e['source'],stream_sha256=hashlib.sha256(Path(e['pbc']).read_bytes()).hexdigest()) for e in app.entries])
(OUT/(name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
root.mainloop()
