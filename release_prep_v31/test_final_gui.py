import ctypes as ct
import json
from pathlib import Path
import sys
import time
import tkinter as tk
from PIL import Image,ImageTk
ct.windll.user32.SetProcessDpiAwarenessContext(ct.c_void_p(-4))
HERE=Path(sys.argv[2]) if "--features" in sys.argv else Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pbc_gui as g
import pbc_windows as win
root=tk.Tk()
root.withdraw()
app=g.App(root)
errors=[]
g.messagebox.showerror=lambda title,text,**kw:errors.append(str(text))
def wait():
    started=time.perf_counter()
    while app.busy or app.pending_encode or (app.warmup_future and not app.warmup_done):
        root.update();time.sleep(.01)
        assert time.perf_counter()-started<120
    root.update()
image=Image.open(HERE/'artifacts/probe-candidate/RGBA.png').copy()
app.start_warmup()
app.loaded("Clipboard image.png",image,source_file=False)
app.start_encode()
wait()
assert not errors,errors
assert app.warmup_info['native']
assert len(app.entries)==1
entry=app.results['encode']
assert not entry['source_file']
expected=image.width*image.height*4/entry['bytes']
assert entry['ratio']==expected
assert app.encode_metrics.ratio.get()==f'{expected:.0f}×'
output=HERE/'artifacts/final-corrections/clipboard encoded.pbc'
g.export(entry,output,'pbc')
app.path_vars['decode'].set(str(output))
app.start_decode();wait()
assert not errors,errors
assert app.decode_metrics.values['bpp'].get()==f"{len(output.read_bytes())*8/(image.width*image.height):.3f}"
for metrics in (app.encode_metrics,app.decode_metrics):
 for field in metrics.fields.values():
  before=field.get();field.insert(0,'changed');assert field.get()==before
app.copy_image('decode');wait()
pixels=win.paste()
with Image.open(app.results['decode']['png']) as expected_image:
 assert pixels.size==expected_image.size and pixels.tobytes()==expected_image.convert('RGBA').tobytes()
assert win.user.IsClipboardFormatAvailable(17)
assert win.user.IsClipboardFormatAvailable(8)
root.clipboard_clear();root.clipboard_append('text only');root.update()
app.paste_clipboard();wait()
assert 'no suitable' in app.orb_status.get()
win.copy_image(image,root.winfo_id())
app.tabs.select(app.encode_tab)
root.update()
app.paste_clipboard();wait()
assert not app.source_is_file and app.input_image.tobytes()==image.tobytes()
app.start_encode();wait()
assert not errors,errors
app.open_viewer('encode');root.update()
assert app.viewer and app.viewer_preview.picture.size==image.size
app.close_viewer();root.update()
assert app.viewer is None
bad=HERE/'artifacts/final-corrections/not image.txt';bad.write_text('not an image')
app.accept_files('encode',[str(bad)]);wait()
assert errors;errors.clear()
app.accept_files('decode',[str(bad)])
assert 'Choose a .pbc' in app.decode_status.get()
app.accept_files('decode',[str(output)]);wait()
assert app.path_vars['decode'].get()==str(output)
root.deiconify();root.update()
calls=[]
g.filedialog.askopenfilename=lambda **kwargs:calls.append(kwargs) or ""
for panel in (app.source_panel,app.decode_input_panel):
 panel.event_generate('<Button-1>')
 panel.body.event_generate('<Button-1>')
 for child in panel.body.winfo_children():
  if isinstance(child,tk.Label):child.event_generate('<Button-1>')
for child in app.source_footer.winfo_children():child.event_generate('<Button-1>')
root.update()
assert not calls
for tab,preview in ((app.encode_tab,app.input_preview),(app.decode_tab,app.decode_source)):
 app.tabs.select(tab);root.update()
 preview.event_generate('<Button-1>');root.update()
assert len(calls)==2
assert app.input_preview.bind('<Return>') and app.decode_source.bind('<space>')
assert app.input_info.get().find('32')==-1
assert app.encode_preview.hint==""
small=HERE/'artifacts/final-corrections/drop fixture Çığ 日本.png'
image.save(small)
win.user.SendMessageW.argtypes=[ct.c_void_p,ct.c_uint,ct.c_size_t,ct.c_ssize_t]
win.user.SendMessageW.restype=ct.c_ssize_t
for operation,path,widget in (("encode",small,app.input_preview),("decode",output,app.decode_source)):
 import struct
 app.tabs.select(app.encode_tab if operation=='encode' else app.decode_tab);root.update()
 payload=struct.pack('<IiiII',20,0,0,0,1)+(str(path)+'\0\0').encode('utf-16-le')
 handle=win.kernel.GlobalAlloc(2,len(payload))
 address=win.kernel.GlobalLock(handle);ct.memmove(address,payload,len(payload));win.kernel.GlobalUnlock(handle)
 win.user.SendMessageW(widget.winfo_id(),0x233,handle,0)
 root.update();time.sleep(.04);root.update();wait()
 assert app.path_vars[operation].get()==str(path)
for scale in (100,125,150):
 root.tk.call('tk','scaling',96*scale/100/72)
 import pbc_ui
 pbc_ui.SCALE=scale/100
 button=g.Button(root,'Test',lambda:None,variant='circle',width=108)
 button.place(x=0,y=0)
 root.update()
 rgba=ImageTk.getimage(button.circle)
 alpha=rgba.getchannel('A')
 assert len(set(alpha.getdata())-{0,255})>30
 button.keyboard_focus=False;button.focus_set();button.draw()
 assert not button.keyboard_focus
 button.invoke(keyboard=True);assert button.keyboard_focus
 button.destroy()
root.withdraw()
data=dict(warmup=app.warmup_info,ratio_exact=True,decode_bpp=True,statistics_readonly=True,
    clipboard_rgba_exact=True,bitmap_formats=True,paste_memory_source=True,
    fullscreen_original_size=True,unsupported_rejected=True,pbc_selection_validated=True,
    antialiased_scales=[100,125,150],keyboard_focus_solid=True,click_regions_bounded=True,
    native_drop_message_routes=True,removed_labels=True)
phase='frozen' if getattr(sys,'frozen',False) else 'source'
(HERE/f'artifacts/final-corrections/features-{phase}.json').write_text(json.dumps(data,indent=2)+'\n')
app.close()
print('Final UI feature checks pass')
