import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
ct.windll.user32.SetProcessDpiAwarenessContext(ct.c_void_p(-4))
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('phase',choices=('before','after'))
p.add_argument('--repeat',type=int,default=3)
p.add_argument('--root',type=Path)
a=p.parse_args()
if a.root:HERE=a.root
sys.path.insert(0,str(HERE/'source'))
sys.path.insert(0,str(HERE/'artifacts/resume-finalization/baseline' if a.phase=='before' else HERE))
import pbc_gui as g
from PBC3 import PBC3
root=g.tk.Tk()
root.withdraw()
app=g.App(root)
errors=[]
g.messagebox.showerror=lambda title,text,**kw:errors.append(str(text))
root.deiconify()
while not app.warmup_done:
 root.update();time.sleep(.005)
active=None

def timed(owner,name,label):
 original=getattr(owner,name)
 def call(*args,**kwargs):
  start=time.perf_counter()
  try:return original(*args,**kwargs)
  finally:
   if active is not None:
    active[label]=active.get(label,0)+time.perf_counter()-start
    active[label+'_calls']=active.get(label+'_calls',0)+1
 setattr(owner,name,call)
for name in ('compress','decompress'):timed(PBC3,name,name)
for name in ('stage','loaded','add_result','card','display_result'):timed(app,name,name)
for name in ('save','thumbnail'):timed(g.Image.Image,name,name)
timed(g.ImageTk,'PhotoImage','photo_image')
timed(g.Preview,'render','preview_render')
original_submit=app.submit

def submit(function,callback,text,**kwargs):
 if not text.startswith('Compress'):return original_submit(function,callback,text,**kwargs)
 def work():
  active['worker_begin']=time.perf_counter()-active['event']
  result=function()
  active['worker_end']=time.perf_counter()-active['event']
  return result
 def completed(result):
  active['callback_begin']=time.perf_counter()-active['event']
  callback(result)
  active['callback_end']=time.perf_counter()-active['event']
 return original_submit(work,completed,text,**kwargs)
app.submit=submit
rows=[]
for filename in ('cat.jpg','flower.jpg'):
 path=HERE.parent/'hpt_data'/filename
 app.loaded(str(path),g.load_image(path))
 root.update()
 for repetition in range(a.repeat):
  active=dict(image=filename,repetition=repetition,event=time.perf_counter())
  app.start_encode()
  while app.busy or app.pending_encode:
   root.update();time.sleep(.001)
  root.update_idletasks()
  ct.windll.dwmapi.DwmFlush()
  active['displayed']=time.perf_counter()-active.pop('event')
  entry=app.results['encode']
  active['reported_codec']=entry['seconds']
  active['queue_delivery']=active['callback_begin']-active['worker_end']
  active['stream_sha256']=hashlib.sha256(Path(entry['pbc']).read_bytes()).hexdigest()
  with g.Image.open(entry['png']) as image:active['pixels_sha256']=hashlib.sha256(image.tobytes()).hexdigest()
  rows.append(active)
  print(json.dumps(active),flush=True)
  active=None
assert not errors,errors
app.close()
(HERE/f'artifacts/resume-finalization/latency-{a.phase}{"-frozen" if getattr(sys,"frozen",False) else ""}.json').write_text(json.dumps(rows,indent=2)+'\n')
