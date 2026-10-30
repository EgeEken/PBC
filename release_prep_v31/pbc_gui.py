import time
GUI_STARTED=time.perf_counter()
import os
from pathlib import Path
import sys

if getattr(sys, "frozen", False):
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PBC/3.1.0/numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "pbc3_cache.FrozenCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(name, "1")
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import math
import queue
import tempfile
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk
from pbc3_types import PBC3Config

BG = "#0d0d0f"
SURFACE = "#161618"
TEXT = "#ededf0"
MUTED = "#a6a6ad"
RED = "#ff3b41"
PRESETS = ("compression", "balanced", "quality", "high_quality")
HIDDEN = {"learned_filler_model_path", "compute_final_mse", "debug_mode", "debug_print", "debug_path", "reuse_selected_delta"}
SEMI = {"patch_count", "auto_downsample_max_pixels", "learned_filler_q"}
CHOICES = {"color_space": ("YCbCr", "RGB"), "channel_cycle": ("Sum", "Mod"),
           "residual_projection_mode": ("bicubic", "cell_mean"), "cell_selection_mode": ("gradient", "hybrid")}
MAX_PIXELS = 32_000_000

def configuration(preset, mode, values):
    config = getattr(PBC3Config, preset)()
    if mode == "Auto":
        return config
    allowed = SEMI if mode == "Semi-Auto" else {f.name for f in dataclasses.fields(config)} - HIDDEN
    for name in allowed:
        value = values[name]
        current = getattr(config, name)
        try:
            value = (str(value).lower() in ("true", "1")) if isinstance(current, bool) else type(current)(value)
        except (TypeError, ValueError):
            raise ValueError(name.replace("_", " ") + ": enter a valid number") from None
        if name in CHOICES and value not in CHOICES[name]:
            raise ValueError("Invalid " + name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(name + " must be finite")
            if name in ("downsample_rate", "warmup_ratio"):
                valid = value == -1 or (0 < value <= 1 if name == "downsample_rate" else 0 <= value <= 1)
            elif name.startswith("q_") or name.startswith("search_q_") or name in ("learned_filler_q", "representability_threshold"):
                valid = 0 <= value <= 1
            elif name == "quality_target_mae":
                valid = 0 <= value <= 255
            elif name == "random_seed":
                valid = 0 <= value < 2**32
            elif name == "mask_size":
                valid = 1 <= value <= 1023
            elif "palette_bitcount" in name:
                valid = 1 <= value <= 9
            elif "max_pixels" in name:
                valid = (0 if name.startswith("warm_") else 1) <= value <= MAX_PIXELS
            else:
                valid = (0 if name == "patch_count" else 1) <= value <= (65535 if name.endswith("_size") else 10000)
            if not valid:
                raise ValueError("Out of range: " + name.replace("_", " "))
        setattr(config, name, value)
    for lower, upper in (("min_patch_size", "max_patch_size"), ("min_cell_size", "max_cell_size")):
        if getattr(config, lower) > getattr(config, upper):
            raise ValueError(lower.replace("_", " ") + " exceeds " + upper.replace("_", " "))
    config.compute_final_mse = True
    config.__post_init__()
    return config

def load_image(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS or max(image.size) > 65535:
            raise ValueError("Image exceeds the 32 megapixel or 65535 pixel dimension limit")
        image = ImageOps.exif_transpose(image)
        alpha = image.mode in ("RGBA", "LA", "PA") or image.mode == "P" and "transparency" in image.info
        return image.convert("RGBA" if alpha else "RGB")

def encode(image, config):
    from PBC3 import PBC3
    from pbc3_native import NativeMSE, select_metric
    backend = select_metric("auto", lambda a, b: None)
    start = time.perf_counter()
    result = PBC3.compress(image, config=config)
    seconds = time.perf_counter() - start
    reconstructed = result.image
    return dict(image=reconstructed, data=result.data, seconds=seconds, mse=result.mse,
                bpp=8 * len(result.data) / (image.width * image.height),
                backend="Native C" if isinstance(backend, NativeMSE) else "Reference",
                config=dataclasses.asdict(config))

def decode(path):
    from pbc3_decoder import decode as read_image, read_file
    data = read_file(path)
    start = time.perf_counter()
    image = read_image(data)
    seconds = time.perf_counter() - start
    return dict(image=image, data=data, seconds=seconds,
                bpp=8 * len(data) / (image.width * image.height))

def export(entry, target, kind):
    target = Path(target)
    if entry.get("source_file",True) and target.resolve() == Path(entry["source"]).resolve():
        raise ValueError("Choose a different output path from the source")
    original = entry["pbc"] if kind == "pbc" else entry["png"]
    with target.open("wb") as output:
        output.write(Path(original).read_bytes())

from pbc_ui import BG, SURFACE, SURFACE2, BORDER, BORDER2, DIM, MUTED, TEXT, RED, RED_DIM, BODY, MONO, px, Button, Surface, Preview, Stats, Pages

class App:
    def __init__(self, root):
        import pbc_ui as ui
        self.root = root
        ui.SCALE = root.winfo_fpixels("1i") / 96
        root.title("PBC V3.1")
        root.geometry(f"{px(1080)}x{px(740)}")
        root.minsize(px(820), px(650))
        root.configure(bg=BG)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.encoder_executor = ThreadPoolExecutor(max_workers=1)
        self.warmup_future = None
        self.warmup_done = False
        self.warmup_info = None
        self.pending_encode = None
        self.drop_queue = queue.Queue()
        self.drop_targets = []
        self.source_is_file = True
        self.viewer = None
        self.active_operation = "encode"
        self.queue = queue.Queue()
        self.storage = tempfile.TemporaryDirectory(prefix="PBC V3.1 ")
        self.entries, self.results = [], {}
        self.registry_cards = {}
        self.selected_id = None
        self.input_image = None
        self.busy = False
        self.counter = self.heartbeats = 0
        self.style()
        header = tk.Frame(root,bg=BG,height=px(62))
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = tk.Frame(header,bg=BG)
        brand.pack(side="left",padx=px(28))
        dot = tk.Canvas(brand,width=px(10),height=px(10),bg=BG,highlightthickness=0)
        dot.create_oval(1,1,px(9),px(9),fill=RED,outline="")
        dot.pack(side="left",padx=(0,px(9)))
        tk.Label(brand,text="PBC",bg=BG,fg=TEXT,font=("Segoe UI Semibold",16)).pack(side="left")
        tk.Label(brand,text="V3.1",bg=BG,fg=DIM,font=MONO).pack(side="left",padx=px(10))
        nav = tk.Frame(header,bg=BG)
        nav.pack(side="right",padx=px(24),fill="y")
        self.navigation = {}
        tk.Frame(root,bg=BORDER,height=1).pack(fill="x")
        self.status = tk.StringVar(value="")
        main = tk.Frame(root,bg=BG)
        main.pack(fill="both",expand=True)
        self.tabs = Pages(main,takefocus=False)
        self.tabs.place(relx=.5,y=0,relheight=1,anchor="n",width=px(1024))
        main.bind("<Configure>",self.resize_content)
        self.encode_tab = tk.Frame(self.tabs,bg=BG,pady=px(20))
        self.decode_tab = tk.Frame(self.tabs,bg=BG,pady=px(20))
        self.registry_tab = tk.Frame(self.tabs,bg=BG,pady=px(20))
        for tab,name in ((self.encode_tab,"Encode"),(self.decode_tab,"Decode"),(self.registry_tab,"Registry")):
            self.tabs.add(tab,text=name)
            button = Button(nav,name,lambda t=tab:self.tabs.select(t),width=84,variant="nav")
            button.pack(side="left",padx=px(4),fill="y",pady=px(8))
            self.navigation[name] = button
        self.tabs.bind("<<NotebookTabChanged>>",self.navigation_changed)
        self.path_vars = {name:tk.StringVar() for name in ("encode","decode")}
        self.make_encode()
        self.make_decode()
        self.make_registry()
        self.navigation_changed()
        root.bind("<MouseWheel>",self.scroll_content,add="+")
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.bind("<Control-v>",self.paste_clipboard,add="+")
        root.bind("<Control-V>",self.paste_clipboard,add="+")
        root.bind("<Control-Tab>",self.next_tab,add="+")
        root.bind("<Escape>",lambda e:self.close_viewer())
        root.bind("<Map>",self.visible,add="+")
        root.after_idle(self.install_drops)
        root.after(25,self.poll)

    def style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".",background=BG,foreground=TEXT,font=BODY,
            bordercolor=BORDER,lightcolor=BORDER,darkcolor=BORDER)
        style.configure("Hidden.TNotebook",background=BG,borderwidth=0,tabmargins=0)
        style.layout("Hidden.TNotebook.Tab",[])
        style.configure("TEntry",fieldbackground=SURFACE,foreground=TEXT,insertcolor=TEXT,
            padding=(px(9),px(7)),borderwidth=1,selectbackground=BORDER2,selectforeground=TEXT)
        style.map("TEntry",bordercolor=[("focus",RED_DIM)],lightcolor=[("focus",RED_DIM)],
            darkcolor=[("focus",RED_DIM)])
        style.configure("TCombobox",fieldbackground=SURFACE,background=SURFACE2,
            foreground=TEXT,arrowcolor=MUTED,padding=px(6),borderwidth=1)
        style.map("TCombobox",fieldbackground=[("readonly",SURFACE)],
            foreground=[("readonly",TEXT)],bordercolor=[("focus",RED_DIM)])
        style.configure("Vertical.TScrollbar",background=BORDER2,troughcolor=BG,
            borderwidth=0,arrowsize=0,width=px(8))
        style.map("Vertical.TScrollbar",background=[("active",DIM)])
        style.configure("Horizontal.TProgressbar",background=RED,troughcolor=SURFACE,
            borderwidth=0,thickness=px(3))
        self.root.option_add("*TCombobox*Listbox.background",SURFACE)
        self.root.option_add("*TCombobox*Listbox.foreground",TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground",BORDER2)
        self.root.option_add("*TCombobox*Listbox.selectForeground",TEXT)

    def resize_content(self,event):
        self.tabs.place_configure(width=min(event.width-px(48),px(1160)))
        if hasattr(self,"settings_canvas"):
            self.settings_canvas.configure(height=px(96 if event.height<px(610) else 125))

    def navigation_changed(self, event=None):
        if not self.tabs.select():
            return
        name = self.tabs.tab(self.tabs.select(),"text")
        for title,button in self.navigation.items():
            button.configure(selected=title==name)

    def title(self, parent, text, detail):
        heading = tk.Frame(parent,bg=BG)
        heading.pack(fill="x",pady=(0,px(16)))
        tk.Label(heading,text=text,bg=BG,fg=TEXT,font=("Segoe UI Semibold",16)).pack(side="left")
        if detail:
            tk.Label(heading,text=detail,bg=BG,fg=DIM,font=MONO).pack(side="right",anchor="s",pady=px(3))
        return heading

    def panel(self, parent, title, column):
        surface = Surface(parent)
        surface.grid(row=0,column=column,sticky="nsew")
        body = surface.body
        body.columnconfigure(0,weight=1)
        body.rowconfigure(1,weight=1)
        tk.Label(body,text=title.upper(),font=MONO,bg=SURFACE,fg=DIM,anchor="w").grid(
            row=0,column=0,sticky="ew",pady=(0,px(12)))
        return surface

    def make_encode(self):
        parent = self.encode_tab
        self.title(parent,"Encode image","")
        self.parameters = tk.Frame(parent,bg=BG)
        self.parameters.pack(side="bottom",fill="x",pady=(px(18),0))
        workflow = tk.Frame(parent,bg=BG)
        workflow.pack(fill="both",expand=True)
        workflow.columnconfigure((0,2),weight=1,uniform="image")
        workflow.columnconfigure(1,minsize=px(132))
        workflow.rowconfigure(0,weight=1)
        self.source_panel = self.panel(workflow,"Source image",0)
        body = self.source_panel.body
        self.input_preview = Preview(body,"Choose a source image","PNG · JPG · WEBP",lambda:self.browse("encode"),source=True)
        self.input_preview.grid(row=1,column=0,sticky="nsew")
        source_footer=tk.Frame(body,bg=SURFACE)
        source_footer.grid(row=2,column=0,sticky="ew",pady=(px(12),px(8)))
        source_footer.columnconfigure(0,weight=1)
        self.input_filename=tk.StringVar(value="No image selected")
        tk.Label(source_footer,textvariable=self.input_filename,bg=SURFACE,fg=MUTED,font=BODY,anchor="w").grid(
            row=0,column=0,sticky="ew",pady=(px(5),px(3)))
        self.input_info = tk.StringVar(value="")
        tk.Label(source_footer,textvariable=self.input_info,bg=SURFACE,fg=DIM,font=MONO,anchor="w").grid(
            row=1,column=0,sticky="ew",pady=(0,px(12)))
        action = tk.Frame(workflow,bg=BG)
        action.grid(row=0,column=1)
        self.compress_button = Button(action,"Compress",self.start_encode,width=108,variant="circle")
        self.compress_button.pack()
        self.compress_button.configure(state="disabled")
        self.orb_status = tk.StringVar(value="")
        self.encode_status_label=tk.Label(action,textvariable=self.orb_status,bg=BG,fg=DIM,font=MONO,
            wraplength=px(124),justify="center")
        self.encode_status_label.pack(pady=(px(10),0))
        self.output_panel = self.panel(workflow,"Reconstructed image",2)
        body = self.output_panel.body
        self.encode_preview = Preview(body,"Your reconstruction appears here","")
        self.encode_preview.grid(row=1,column=0,sticky="nsew")
        output_footer=tk.Frame(body,bg=SURFACE)
        output_footer.grid(row=2,column=0,sticky="ew",pady=(px(12),px(8)))
        output_footer.columnconfigure(0,weight=1)
        self.encode_metrics = Stats(output_footer,("Size","bpp","MSE","Time"),ratio=True)
        self.encode_metrics.grid(row=0,column=0,sticky="ew",pady=(0,px(12)))
        saves = tk.Frame(output_footer,bg=SURFACE)
        saves.grid(row=1,column=0,sticky="ew")
        saves.columnconfigure((0,1),weight=1)
        self.encode_save = Button(saves,"Save PBC",lambda:self.save("encode","pbc"),width=110,state="disabled")
        self.encode_save.grid(row=0,column=0,sticky="ew",padx=(0,px(4)))
        self.encode_png = Button(saves,"Save PNG",lambda:self.save("encode","png"),width=110,state="disabled")
        self.encode_png.grid(row=0,column=1,sticky="ew",padx=(px(4),0))
        self.source_footer=source_footer
        self.output_footer=output_footer
        source_footer.bind("<Configure>",self.align_footers)
        output_footer.bind("<Configure>",self.align_footers)
        self.output_interactions(self.encode_preview,"encode")
        self.encode_stats = tk.StringVar()
        self.preset = tk.StringVar(value="quality")
        self.mode = tk.StringVar(value="Auto")
        tk.Frame(self.parameters,bg=BORDER,height=1).pack(fill="x",pady=(0,px(14)))
        head = tk.Frame(self.parameters,bg=BG)
        head.pack(fill="x",pady=(0,px(10)))
        tk.Label(head,text="PARAMETERS",bg=BG,fg=DIM,font=MONO).pack(side="left")
        segments = tk.Frame(head,bg=SURFACE,highlightthickness=1,highlightbackground=BORDER)
        segments.pack(side="right")
        self.mode_buttons = {}
        for name in ("Auto","Semi-Auto","Manual"):
            button = Button(segments,name,lambda n=name:self.mode.set(n),width=86,variant="segment")
            button.pack(side="left",padx=px(2),pady=px(2))
            self.mode_buttons[name] = button
        presets = tk.Frame(self.parameters,bg=BG)
        presets.pack(fill="x")
        self.preset_buttons = {}
        for index,name in enumerate(PRESETS):
            presets.columnconfigure(index,weight=1)
            button = Button(presets,name.replace("_"," ").title(),lambda n=name:self.preset.set(n),width=120,variant="preset")
            button.grid(row=0,column=index,sticky="ew",padx=(0 if index==0 else px(4),0 if index==3 else px(4)))
            self.preset_buttons[name] = button
        self.settings = tk.Frame(self.parameters,bg=BG)
        self.settings_canvas = tk.Canvas(self.settings,bg=BG,highlightthickness=0,height=px(125))
        self.settings_canvas.pack(side="left",fill="both",expand=True)
        scrollbar = ttk.Scrollbar(self.settings,orient="vertical",command=self.settings_canvas.yview)
        scrollbar.pack(side="right",fill="y",padx=(px(8),0))
        self.settings_canvas.configure(yscrollcommand=scrollbar.set)
        self.editor = tk.Frame(self.settings_canvas,bg=BG)
        self.editor_window = self.settings_canvas.create_window(0,0,anchor="nw",window=self.editor)
        self.editor.bind("<Configure>",lambda e:self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind("<Configure>",lambda e:self.settings_canvas.itemconfigure(self.editor_window,width=e.width))
        self.parameter_vars = {}
        self.mode.trace_add("write",lambda *a:self.refresh_parameters())
        self.preset.trace_add("write",lambda *a:self.refresh_parameters())
        self.refresh_parameters()

    def refresh_parameters(self):
        for name,button in self.mode_buttons.items():
            button.configure(selected=name==self.mode.get())
        for name,button in self.preset_buttons.items():
            button.configure(selected=name==self.preset.get())
        for child in self.editor.winfo_children():
            child.destroy()
        self.parameter_vars = {}
        mode = self.mode.get()
        if mode=="Auto":
            self.settings.pack_forget()
            return
        self.settings.pack(fill="x",pady=(px(10),0))
        config = getattr(PBC3Config,self.preset.get())()
        groups = (
            ("Budget & search",("patch_count","search_depth","proposal_depth","exact_depth","top_k","search_q_start","search_q_end","anchor_block_size")),
            ("Patch geometry",("min_patch_size","max_patch_size","min_cell_size","max_cell_size","cell_sizes_per_candidate","residual_projection_mode","cell_selection_mode","representability_threshold")),
            ("Initialization",("auto_downsample_init","init_search_depth","downsample_init_cell_size","downsample_palette_bitcount","downsample_rate","auto_downsample_max_pixels","warmup_ratio","warm_downsample_max_pixels")),
            ("Color & palette",("color_space","channel_cycle","q_init","q_start","q_end","patch_palette_bitcount","mask_size","positive_bias","quality_target_mae")),
            ("Learned filler & output",("learned_filler_enabled","learned_filler_top_k","learned_filler_q","learned_filler_candidates","use_lzma","random_seed")))
        row = 0
        for title,names in groups:
            names = [name for name in names if name not in HIDDEN and (mode=="Manual" or name in SEMI)]
            if not names:
                continue
            if mode=="Manual":
                tk.Label(self.editor,text=title.upper(),bg=BG,fg=DIM,font=("Consolas",9),anchor="w").grid(
                    row=row,column=0,columnspan=4,sticky="ew",pady=(px(7),px(5)))
                row+=1
            for index,name in enumerate(names):
                column=index%2*2
                r=row+index//2
                tk.Label(self.editor,text=name.replace("_"," ").capitalize(),bg=BG,fg=MUTED,font=("Segoe UI",9),
                    anchor="w").grid(row=r,column=column,sticky="w",padx=(0,px(12)),pady=px(3))
                value = getattr(config,name)
                var=tk.StringVar(value=str(value))
                self.parameter_vars[name]=var
                if name in CHOICES or isinstance(value,bool):
                    widget=ttk.Combobox(self.editor,textvariable=var,values=CHOICES.get(name,("True","False")),
                        state="readonly",width=13,font=MONO)
                    widget.bind("<MouseWheel>",self.scroll_content)
                else:
                    widget=ttk.Entry(self.editor,textvariable=var,width=13,font=MONO)
                widget.grid(row=r,column=column+1,sticky="ew",padx=(0,px(18) if column==0 else 0),pady=px(3))
                widget.bind("<FocusIn>",self.focus_parameter,add="+")
            row+=(len(names)+1)//2
        self.editor.columnconfigure((1,3),weight=1)
        self.settings_canvas.yview_moveto(0)

    def focus_parameter(self,event):
        self.editor.update_idletasks()
        y=event.widget.winfo_y()
        top=self.settings_canvas.canvasy(0)
        height=self.settings_canvas.winfo_height()
        if y<top or y+event.widget.winfo_height()>top+height:
            self.settings_canvas.yview_moveto(max(0,y-height/2)/max(1,self.editor.winfo_height()))

    def align_footers(self,event=None):
        height=max(self.source_footer.winfo_reqheight(),self.output_footer.winfo_reqheight())+px(20)
        for panel in (self.source_panel,self.output_panel):
            panel.body.rowconfigure(2,minsize=height)

    def make_decode(self):
        parent=self.decode_tab
        self.title(parent,"Decode image","")
        workflow=tk.Frame(parent,bg=BG)
        workflow.pack(fill="both",expand=True)
        workflow.columnconfigure((0,2),weight=1,uniform="decode")
        workflow.columnconfigure(1,minsize=px(132))
        workflow.rowconfigure(0,weight=1)
        self.decode_input_panel=self.panel(workflow,"Compressed file",0)
        self.decode_input_panel.configure(height=px(190))
        self.decode_input_panel.grid_configure(sticky="ew")
        body=self.decode_input_panel.body
        self.decode_source=Preview(body,"Choose a .pbc file","Click, drop, or paste a copied file",
            lambda:self.browse("decode"),source=True)
        self.decode_source.grid(row=1,column=0,sticky="nsew")
        self.decode_file_info=tk.StringVar(value="")
        action=tk.Frame(workflow,bg=BG)
        action.grid(row=0,column=1)
        self.decode_button=Button(action,"Decode",self.start_decode,width=108,variant="circle",state="disabled")
        self.decode_button.pack()
        self.decode_status=tk.StringVar(value="")
        self.decode_status_label=tk.Label(action,textvariable=self.decode_status,bg=BG,fg=DIM,font=MONO,
            wraplength=px(124),justify="center")
        self.decode_status_label.pack(pady=(px(10),0))
        self.decode_output_panel=self.panel(workflow,"Reconstructed image",2)
        body=self.decode_output_panel.body
        self.decode_preview=Preview(body,"Your decoded image appears here","")
        self.decode_preview.grid(row=1,column=0,sticky="nsew")
        self.output_interactions(self.decode_preview,"decode")
        self.decode_metrics=Stats(body,("File size","bpp","Time"))
        self.decode_metrics.grid(row=2,column=0,sticky="ew",pady=(px(12),px(14)))
        self.decode_png=Button(body,"Save PNG",lambda:self.save("decode","png"),width=130,state="disabled")
        self.decode_png.grid(row=3,column=0,sticky="ew",pady=(0,px(8)))
        self.decode_stats=tk.StringVar()
        self.path_vars["decode"].trace_add("write",lambda *a:self.decode_metadata())

    def decode_metadata(self,info=None):
        path=Path(self.path_vars["decode"].get())
        try:
            size=path.stat().st_size
        except OSError:
            return
        self.decode_source.placeholder=path.name
        self.decode_source.hint=f"{size/1024:.1f} KB"
        if info:
            self.decode_source.hint+=f" · {info['width']} × {info['height']}"
        self.decode_source.render()

    def make_registry(self):
        self.registry_count=tk.StringVar(value="0 results · session only")
        heading=self.title(self.registry_tab,"Registry","")
        tk.Label(heading,textvariable=self.registry_count,bg=BG,fg=DIM,font=MONO).pack(side="right")
        area=tk.Frame(self.registry_tab,bg=BG)
        area.pack(fill="both",expand=True)
        self.registry_canvas=tk.Canvas(area,bg=BG,highlightthickness=0,bd=0)
        self.registry_canvas.pack(side="left",fill="both",expand=True)
        scrollbar=ttk.Scrollbar(area,orient="vertical",command=self.registry_canvas.yview)
        scrollbar.pack(side="right",fill="y",padx=(px(10),0))
        self.registry_canvas.configure(yscrollcommand=scrollbar.set)
        self.registry_body=tk.Frame(self.registry_canvas,bg=BG)
        self.registry_window=self.registry_canvas.create_window(0,0,anchor="nw",window=self.registry_body)
        self.registry_body.bind("<Configure>",lambda e:self.registry_canvas.configure(scrollregion=self.registry_canvas.bbox("all")))
        self.registry_canvas.bind("<Configure>",self.resize_registry)
        self.registry_empty=tk.Frame(self.registry_canvas,bg=BG)
        self.registry_empty.place(relx=.5,rely=.42,anchor="center")
        tk.Label(self.registry_empty,text="No results yet",bg=BG,fg=MUTED,font=("Segoe UI",14)).pack()
        tk.Label(self.registry_empty,text="Compressed and decoded images appear here.",bg=BG,fg=DIM,font=MONO).pack(pady=px(10))
        Button(self.registry_empty,"Encode an image",lambda:self.tabs.select(self.encode_tab),width=150).pack(pady=px(8))

    def resize_registry(self,event):
        self.registry_canvas.itemconfigure(self.registry_window,width=event.width)
        self.arrange_registry()

    def arrange_registry(self):
        columns=2 if self.registry_canvas.winfo_width()>=px(880) else 1
        for c in (0,1):
            self.registry_body.columnconfigure(c,weight=1 if c<columns else 0)
        for i,entry in enumerate(reversed(self.entries)):
            self.registry_cards[entry["id"]].grid(row=i//columns,column=i%columns,sticky="ew",
                padx=(0,px(12) if columns==2 and i%2==0 else 0),pady=(0,px(12)))

    def card(self,entry):
        card=Surface(self.registry_body,height=174)
        body=card.body
        body.columnconfigure(1,weight=1)
        body.rowconfigure(0,weight=1)
        image=tk.Label(body,image=entry["photo"],bg=SURFACE,cursor="hand2",width=px(110))
        image.grid(row=0,column=0,sticky="nsew",padx=(0,px(16)))
        image.bind("<Double-1>",lambda e:self.reopen(entry))
        info=tk.Frame(body,bg=SURFACE)
        info.grid(row=0,column=1,sticky="nsew")
        tk.Label(info,text=entry["operation"].upper(),bg=SURFACE,fg=DIM,font=("Consolas",9),anchor="w").pack(fill="x")
        title=tk.Label(info,text=(Path(entry["source"]).name if len(Path(entry["source"]).name)<38 else Path(entry["source"]).name[:35]+"…"),bg=SURFACE,fg=TEXT,font=("Segoe UI Semibold",11),anchor="w")
        title.pack(fill="x",pady=(px(3),px(6)))
        title.bind("<Configure>",lambda e:title.configure(wraplength=max(1,e.width-px(6))))
        tk.Label(info,text=entry["card_stats"],bg=SURFACE,fg=MUTED,font=MONO,justify="left",anchor="w").pack(fill="x")
        actions=tk.Frame(body,bg=SURFACE)
        actions.grid(row=1,column=0,columnspan=2,sticky="ew",pady=(px(10),0))
        for c in range(3):
            actions.columnconfigure(c,weight=1)
        for c,(text,command) in enumerate((("Reopen",lambda:self.reopen(entry)),
                ("Save PBC",lambda:self.save("registry","pbc",entry)),
                ("Save PNG",lambda:self.save("registry","png",entry)))):
            Button(actions,text,command,width=90).grid(row=0,column=c,sticky="ew",padx=(0,px(6) if c<2 else 0))
        card.bind("<Enter>",lambda e:card.itemconfigure("border",outline=BORDER2))
        card.bind("<Leave>",lambda e:card.itemconfigure("border",outline=RED_DIM if self.selected_id==entry["id"] else BORDER))
        self.registry_cards[entry["id"]]=card
        self.registry_empty.place_forget()
        self.registry_count.set(f"{len(self.entries)} results · latest 30 · session only")
        self.arrange_registry()

    def scroll_content(self,event):
        widget=str(event.widget)
        if widget.startswith(str(self.settings_canvas)):
            self.settings_canvas.yview_scroll(-int(event.delta/120),"units")
            return "break"
        if widget.startswith(str(self.registry_canvas)):
            self.registry_canvas.yview_scroll(-int(event.delta/120),"units")
            return "break"


    def next_tab(self,event=None):
        pages=self.tabs.tabs()
        self.tabs.select(pages[(pages.index(self.tabs.select())+1)%len(pages)])
        return "break"

    def visible(self,event):
        if event.widget==self.root and self.warmup_future is None:
            self.visible_seconds=time.perf_counter()-GUI_STARTED
            self.root.after(50,self.start_warmup)

    def start_warmup(self):
        if self.warmup_future is not None:
            return
        from pbc_startup import initialize
        self.notify("Preparing encoder…","encode")
        self.compress_button.configure(state="disabled")
        self.warmup_future=self.encoder_executor.submit(initialize)

    def notify(self,text,operation="encode",error=False,transient=False):
        self.active_operation=operation
        self.status.set(text)
        variable=self.orb_status if operation=="encode" else self.decode_status
        label=self.encode_status_label if operation=="encode" else self.decode_status_label
        variable.set(text)
        label.configure(fg=RED if error else DIM)
        if self.viewer and transient:
            self.viewer_notice.set(text)
        if transient:
            self.root.after(2500,lambda:variable.set("") if variable.get()==text else None)

    def install_drops(self):
        from pbc_windows import DropTarget
        def attach(widget,operation):
            self.drop_targets.append(DropTarget(widget,lambda paths:self.drop_queue.put((operation,paths))))
            for child in widget.winfo_children():
                attach(child,operation)
        attach(self.source_panel,"encode")
        attach(self.decode_input_panel,"decode")

    def accept_files(self,operation,paths):
        if self.busy:
            return
        if len(paths)!=1:
            self.notify("Choose one file at a time.",operation,error=True)
            return
        path=str(paths[0])
        if operation=="encode":
            self.load_input(path)
        elif Path(path).suffix.lower()!=".pbc":
            self.notify("Choose a .pbc file.","decode",error=True)
        else:
            def check():
                from pbc3_decoder import inspect,read_file
                return inspect(read_file(path))
            def selected(info):
                self.path_vars["decode"].set(path)
                self.decode_metadata(info)
                self.decode_button.configure(state="normal")
            self.submit(check,selected,"Checking file…",operation="decode")

    def browse(self,operation):
        if self.busy:
            return
        types=[("PBC image","*.pbc")] if operation=="decode" else [
            ("Images","*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif"),("All files","*.*")]
        path=filedialog.askopenfilename(parent=self.root,filetypes=types)
        if path:
            self.accept_files(operation,[path])

    def paste_clipboard(self,event=None):
        widget=self.root.focus_get()
        if isinstance(widget,(tk.Entry,ttk.Entry,ttk.Combobox,tk.Text)):
            if str(widget.cget("state")) not in ("readonly","disabled"):
                return
        operation=self.tabs.tab(self.tabs.select(),"text").lower()
        if operation not in ("encode","decode") or self.busy:
            return
        from pbc_windows import paste
        def pasted(value):
            if isinstance(value,Image.Image) and operation=="encode":
                self.loaded("Clipboard image.png",value,source_file=False)
            elif isinstance(value,list) and value:
                self.root.after_idle(lambda:self.accept_files(operation,value))
            else:
                self.notify("Clipboard has no suitable image or file.",operation,error=True)
        self.submit(paste,pasted,"Reading clipboard…",operation=operation)
        return "break"

    def submit(self,function,callback,text,operation="encode",encoder=False):
        if self.busy:
            return
        self.busy=True
        self.compress_button.configure(state="disabled")
        self.decode_button.configure(state="disabled")
        self.compress_button.set_busy(text.startswith("Compress"))
        self.decode_button.set_busy(text.startswith("Decod"))
        self.job_text=text
        self.notify(text,operation)
        future=(self.encoder_executor if encoder else self.executor).submit(function)
        future.add_done_callback(lambda done:self.queue.put((done,callback,operation)))

    def poll(self):
        self.heartbeats+=1
        if self.warmup_future and self.warmup_future.done() and not self.warmup_done:
            self.warmup_done=True
            try:
                self.warmup_info=self.warmup_future.result()
                if not self.busy:
                    self.notify("","encode")
            except Exception as error:
                self.warmup_info=dict(error=str(error))
                self.notify("Preparation failed. Compress will retry.","encode",error=True)
            if not self.busy:
                self.compress_button.configure(state="normal" if self.input_image else "disabled")
        try:
            operation,paths=self.drop_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.accept_files(operation,paths)
        try:
            future,callback,operation=self.queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy=False
            self.compress_button.set_busy(False)
            self.decode_button.set_busy(False)
            self.compress_button.configure(state="normal" if self.input_image and self.warmup_done else "disabled")
            self.decode_button.configure(state="normal" if self.path_vars["decode"].get() else "disabled")
            self.notify("",operation)
            try:
                callback(future.result())
            except Exception as error:
                text="File not found. Choose an existing file." if isinstance(error,FileNotFoundError) else str(error)
                if operation=="decode" and not isinstance(error,FileNotFoundError):
                    text="Could not read this PBC file. It may be damaged, unsupported, or exceed the supported limits."
                elif isinstance(error,Image.UnidentifiedImageError):
                    text="Choose a supported image file."
                self.notify("Operation failed",operation,error=True)
                messagebox.showerror("PBC",text,parent=self.root)
        if self.pending_encode and self.warmup_done and not self.busy:
            payload=self.pending_encode
            self.pending_encode=None
            self.run_encode(*payload)
        self.root.after(25,self.poll)

    def load_input(self,path=None):
        path=path or self.path_vars["encode"].get().strip()
        if path:
            self.submit(lambda:load_image(path),lambda image:self.loaded(path,image),"Loading image…")

    def loaded(self,path,image,source_file=True):
        if (getattr(self,"loaded_path",None)!=path or not source_file) and "encode" in self.results:
            del self.results["encode"]
            self.encode_preview.show(None)
            for value in self.encode_metrics.values.values():
                value.set("—")
            self.encode_metrics.ratio.set("")
            self.encode_save.configure(state="disabled")
            self.encode_png.configure(state="disabled")
        self.input_image=image
        self.loaded_path=path
        self.source_is_file=source_file
        self.path_vars["encode"].set(path)
        name=Path(path).name
        self.input_filename.set(name if len(name)<42 else name[:39]+"…")
        if self.input_preview.picture is not image:
            self.input_preview.show(image)
        self.compress_button.configure(state="normal" if self.warmup_done else "disabled")
        self.input_info.set(f"{image.width} × {image.height} · {image.mode}")

    def start_encode(self):
        if self.busy:
            return
        path=self.path_vars["encode"].get().strip()
        if not path:
            self.browse("encode")
            return
        try:
            config=configuration(self.preset.get(),self.mode.get(),{k:v.get() for k,v in self.parameter_vars.items()})
        except ValueError as error:
            messagebox.showerror("Configuration",str(error),parent=self.root)
            return
        source=self.input_image if getattr(self,"loaded_path",None)==path else path
        payload=(source,path,config,self.source_is_file if source is self.input_image else True)
        if not self.warmup_done:
            self.start_warmup()
            self.pending_encode=payload
            self.notify("Preparing encoder…","encode")
            return
        self.run_encode(*payload)

    def run_encode(self,source,path,config,source_file):
        def run():
            from pbc_startup import signatures
            image=source if isinstance(source,Image.Image) else load_image(source)
            before=signatures()
            result=self.stage(encode(image,config))
            result["jit_added"]={name:values for name,values in signatures().items() if values!=before.get(name)}
            return image,result
        def completed(pair):
            image,result=pair
            self.loaded(path,image,source_file=source_file)
            self.last_jit_added=result["jit_added"]
            self.add_result("encode",path,result)
        self.submit(run,completed,"Compressing…",encoder=True)

    def start_decode(self):
        if self.busy:
            return
        path=self.path_vars["decode"].get().strip()
        if not path:
            self.browse("decode")
            return
        self.submit(lambda:self.stage(decode(path)),lambda result:self.add_result("decode",path,result),
            "Decoding…",operation="decode")

    def output_interactions(self,preview,operation):
        preview.configure(takefocus=True)
        preview.bind("<Double-Button-1>",lambda e:self.open_viewer(operation))
        preview.bind("<Button-3>",lambda e:self.image_menu(e,operation))
        preview.bind("<Control-c>",lambda e:self.copy_image(operation))

    def image_menu(self,event,operation,entry=None):
        entry=entry or self.results.get(operation)
        if not entry:
            return
        menu=tk.Menu(self.root,tearoff=False,bg=SURFACE2,fg=TEXT,activebackground=BORDER2,
            activeforeground=TEXT,relief="flat",bd=1)
        menu.add_command(label="Copy image",command=lambda:self.copy_image(operation,entry))
        menu.add_command(label="Exit fullscreen" if self.viewer else "View fullscreen",
            command=self.close_viewer if self.viewer else lambda:self.open_viewer(operation,entry))
        self.image_context_menu=menu
        try:
            menu.tk_popup(event.x_root,event.y_root)
        finally:
            menu.grab_release()

    def copy_image(self,operation,entry=None):
        entry=entry or self.results.get(operation)
        if not entry or self.busy:
            return
        owner=self.root.winfo_id()
        def copy():
            from pbc_windows import copy_image
            with Image.open(entry["png"]) as image:
                copy_image(image,owner)
        self.submit(copy,lambda _:self.notify("Image copied",operation,transient=True),
            "Copying image…",operation=operation)

    def open_viewer(self,operation,entry=None):
        entry=entry or self.results.get(operation)
        if not entry or self.viewer:
            return
        preview=getattr(self,operation+"_preview")
        if self.results.get(operation) is entry and preview.picture is not None:
            image=preview.picture
        else:
            with Image.open(entry["png"]) as opened:
                image=opened.copy()
        self.viewer_geometry=self.root.geometry()
        self.viewer_state=self.root.state()
        self.viewer=tk.Frame(self.root,bg=BG)
        self.viewer.place(x=0,y=0,relwidth=1,relheight=1)
        self.viewer.lift()
        header=tk.Frame(self.viewer,bg=BG)
        header.pack(fill="x",padx=px(20),pady=px(12))
        Button(header,"Close ×",self.close_viewer,width=96).pack(side="right")
        self.viewer_notice=tk.StringVar(value="")
        tk.Label(self.viewer,textvariable=self.viewer_notice,bg=BG,fg=MUTED,font=MONO).pack(side="bottom",pady=px(8))
        self.viewer_preview=Preview(self.viewer,"")
        self.viewer_preview.configure(bg=BG,takefocus=True)
        self.viewer_preview.pack(fill="both",expand=True,padx=px(16),pady=(0,px(8)))
        self.viewer_preview.show(image)
        self.viewer_preview.bind("<Double-Button-1>",lambda e:self.close_viewer())
        self.viewer_preview.bind("<Button-3>",lambda e:self.image_menu(e,operation,entry))
        self.viewer_preview.bind("<Control-c>",lambda e:self.copy_image(operation,entry))
        self.root.attributes("-fullscreen",True)
        self.viewer_preview.focus_set()

    def close_viewer(self):
        if self.viewer:
            self.root.attributes("-fullscreen",False)
            self.viewer_preview.picture=None
            self.viewer.destroy()
            self.viewer=None
            if self.viewer_state=="zoomed":
                self.root.state("zoomed")
            else:
                self.root.geometry(self.viewer_geometry)


    def stage(self, result):
        stem = Path(self.storage.name) / uuid.uuid4().hex
        result["png"], result["pbc"] = str(stem.with_suffix(".png")), str(stem.with_suffix(".pbc"))
        result["image"].save(result["png"],compress_level=1,optimize=False)
        Path(result["pbc"]).write_bytes(result["data"])
        return result

    def add_result(self, operation, source, result):
        self.counter += 1
        image = result["image"]
        stats = f"{image.width} × {image.height} · {len(result['data']):,} bytes · {result['seconds']:.3f} s"
        if operation=="encode":
            stats += f"\n{result['bpp']:.4f} bpp · MSE {result['mse']:.3f} · {result['backend']}"
        thumb=image.copy()
        thumb.thumbnail((px(110),px(86)),Image.Resampling.LANCZOS)
        photo=ImageTk.PhotoImage(thumb)
        card_stats=f"{image.width} × {image.height} · {image.mode}\n{len(result['data'])/1024:.1f} KB · {result['seconds']:.3f} s"
        if operation=="encode":
            card_stats+=f"\n{result['bpp']:.3f} bpp · MSE {result['mse']:.2f}"
        entry=dict(id=str(self.counter),operation=operation,source=str(source),png=result["png"],pbc=result["pbc"],
            stats=stats,card_stats=card_stats,photo=photo,seconds=result["seconds"],config=result.get("config"),
            dimensions=f"{image.width} × {image.height} · {image.mode}",bytes=len(result["data"]),
            bpp=result["bpp"],mse=result.get("mse"),backend=result.get("backend"),
            ratio=image.width*image.height*len(image.getbands())/len(result["data"]),
            source_file=self.source_is_file if operation=="encode" else True)
        self.entries.append(entry)
        if len(self.entries)>30:
            old=self.entries.pop(0)
            self.registry_cards.pop(old["id"]).destroy()
            if self.selected_id==old["id"]:
                self.selected_id=None
            for name,current in list(self.results.items()):
                if current is old:
                    del self.results[name]
                    getattr(self,name+"_preview").show(None)
                    if name=="encode":
                        self.encode_save.configure(state="disabled")
                        self.encode_png.configure(state="disabled")
                    else:
                        self.decode_png.configure(state="disabled")
            Path(old["png"]).unlink()
            Path(old["pbc"]).unlink()
        self.card(entry)
        self.display_result(operation,entry,image)

    def display_result(self, operation, entry, image):
        self.results[operation]=entry
        getattr(self,operation+"_preview").show(image)
        getattr(self,operation+"_stats").set(entry["stats"])
        if operation=="encode":
            values={"Size":f"{entry['bytes']/1024:.1f} KB","bpp":f"{entry['bpp']:.3f}",
                "MSE":f"{entry['mse']:.2f}","Time":f"{entry['seconds']:.3f} s"}
            self.encode_metrics.ratio.set(f"{entry['ratio']:.0f}×")
            self.encode_save.configure(state="normal")
            self.encode_png.configure(state="normal")
        else:
            values={"File size":f"{entry['bytes']/1024:.1f} KB","bpp":f"{entry['bpp']:.3f}","Time":f"{entry['seconds']:.3f} s"}
            self.decode_metadata(dict(width=image.width,height=image.height))
            self.decode_png.configure(state="normal")
        for name,value in values.items():
            getattr(self,operation+"_metrics").values[name].set(value)

    def selected(self):
        return next((entry for entry in self.entries if entry["id"]==self.selected_id),None)

    def select_registry(self, item_id=None):
        self.selected_id=item_id or self.selected_id
        for key,card in self.registry_cards.items():
            card.itemconfigure("border",outline=RED_DIM if key==self.selected_id else BORDER)

    def reopen(self, entry=None):
        entry = entry or self.selected()
        if entry:
            self.path_vars[entry["operation"]].set(entry["source"])
            if entry["operation"] == "encode":
                self.input_image=None
                self.loaded_path=None
                self.input_preview.show(None)
                self.input_filename.set(Path(entry["source"]).name)
                self.compress_button.configure(state="disabled" if not entry.get("source_file",True) else "normal")
                self.input_info.set(entry["dimensions"])
            with Image.open(entry["png"]) as image:
                self.display_result(entry["operation"], entry, image.copy())
            self.tabs.select(self.encode_tab if entry["operation"] == "encode" else self.decode_tab)

    def save(self, operation, kind, entry=None):
        entry = entry or (self.selected() if operation=="registry" else self.results.get(operation))
        if not entry:
            return
        extension = "." + kind
        path = filedialog.asksaveasfilename(defaultextension=extension,
            initialfile=Path(entry["source"]).stem + extension,
            filetypes=[("PBC image" if kind == "pbc" else "PNG image", "*"+extension)], confirmoverwrite=True)
        if path:
            try:
                export(entry, path, kind)
                self.notify("Saved",entry["operation"],transient=True)
            except Exception as error:
                messagebox.showerror("Save", str(error), parent=self.root)

    def close(self):
        if self.busy and not messagebox.askyesno("PBC", "An operation is running. Close PBC?", parent=self.root):
            return
        for timer in self.root.tk.call("after","info"):
            self.root.after_cancel(timer)
        for target in self.drop_targets:
            target.close()
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.encoder_executor.shutdown(wait=False,cancel_futures=True)
        self.storage.cleanup()
        self.root.destroy()

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
