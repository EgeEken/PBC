import ctypes as ct
from concurrent.futures import Future
import json
from pathlib import Path
import sys
ct.windll.user32.SetProcessDpiAwarenessContext(ct.c_void_p(-4))
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pbc_gui as g
rows=[]
for scale in (100,125,150):
    root=g.tk.Tk()
    root.withdraw()
    root.tk.call('tk','scaling',96*scale/100/72)
    app=g.App(root)
    app.warmup_done=True
    app.warmup_future=Future()
    app.warmup_future.set_result({})
    root.deiconify()
    root.update()
    for size in ('normal','minimum','maximized'):
        root.state('normal')
        if size=='maximized':
            root.state('zoomed')
        else:
            w,h=(1080,740) if size=='normal' else (820,650)
            root.geometry(f'{g.px(w)}x{g.px(h)}')
        root.update()
        for mode in ('Auto','Semi-Auto','Manual'):
            app.mode.set(mode)
            app.tabs.select(app.encode_tab)
            root.update()
            pads=[]
            for button,panel in ((app.encode_save,app.output_panel),(app.encode_png,app.output_panel)):
                assert button.winfo_ismapped(),(scale,size,mode,'not mapped')
                ancestor=button.master
                while ancestor!=app.encode_tab:
                    assert button.winfo_rooty()>=ancestor.winfo_rooty(),(scale,size,mode,'above ancestor')
                    assert button.winfo_rooty()+button.winfo_height()<=ancestor.winfo_rooty()+ancestor.winfo_height(),(scale,size,mode,'clipped',str(ancestor))
                    ancestor=ancestor.master
                bottom=panel.body.winfo_rooty()+panel.body.winfo_height()-button.winfo_rooty()-button.winfo_height()
                assert bottom>=g.px(8),(scale,size,mode,'bottom padding',bottom)
                pads.append(bottom)
            assert abs(app.input_preview.winfo_height()-app.encode_preview.winfo_height())<=1
            rows.append(dict(scale=scale,size=size,mode=mode,window=[root.winfo_width(),root.winfo_height()],
                preview_height=app.input_preview.winfo_height(),save_bottom_padding=pads))
        app.tabs.select(app.decode_tab)
        root.update()
        button=app.decode_png
        bottom=app.decode_output_panel.body.winfo_rooty()+app.decode_output_panel.body.winfo_height()-button.winfo_rooty()-button.winfo_height()
        assert bottom>=g.px(8),(scale,size,'decode bottom',bottom)
        assert app.decode_input_panel.winfo_height()<app.decode_output_panel.winfo_height()*.6
    app.close()
(HERE/'artifacts/final-corrections/layout-tests.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Layout:',len(rows),'scale/size/mode combinations; Save controls fit their panels with bottom padding')
