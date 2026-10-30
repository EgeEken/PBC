import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

BG = "#0d0d0f"
SURFACE = "#161618"
SURFACE2 = "#1d1d20"
BORDER = "#262629"
BORDER2 = "#323237"
FAINT = "#56565d"
DIM = "#82828a"
MUTED = "#a6a6ad"
TEXT = "#ededf0"
RED = "#ff3b41"
RED_DIM = "#c92d32"
SCALE = 1.0
BODY = ("Segoe UI", 10)
MONO = ("Consolas", 10)

def px(value):
    return round(value * SCALE)

def rounded(canvas, x1, y1, x2, y2, radius, **kwargs):
    r = min(radius, (x2-x1)/2, (y2-y1)/2)
    return canvas.create_polygon(x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
        x2, y2-r, x2, y2, x2-r, y2, x1+r, y2, x1, y2, x1, y2-r,
        x1, y1+r, x1, y1, smooth=True, splinesteps=24, **kwargs)

class Button(tk.Canvas):
    def __init__(self, parent, text, command, width=100, variant="secondary", **kwargs):
        self.text, self.command, self.variant = text, command, variant
        self.hover = self.pressed = self.selected = self.busy = False
        self.keyboard_focus = True
        self.phase = 0
        self.timer = None
        super().__init__(parent, width=px(width), height=px(108 if variant=="circle" else 34),
            bg=parent.cget("background"), highlightthickness=0, bd=0, takefocus=kwargs.get("state")!="disabled",
            cursor="arrow" if kwargs.get("state")=="disabled" else "hand2", **kwargs)
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self.change(hover=True))
        self.bind("<Leave>", lambda e: self.change(hover=False, pressed=False))
        self.bind("<ButtonPress-1>", lambda e: self.change(pressed=True, keyboard_focus=False))
        self.bind("<ButtonRelease-1>", self.release)
        self.bind("<FocusIn>", lambda e: self.draw())
        self.bind("<FocusOut>", lambda e: self.change(keyboard_focus=True))
        self.bind("<Return>", lambda e: self.invoke(keyboard=True))
        self.bind("<space>", lambda e: self.invoke(keyboard=True))

    def configure(self, cnf=None, **kwargs):
        if "text" in kwargs:
            self.text = kwargs.pop("text")
        if "selected" in kwargs:
            self.selected = kwargs.pop("selected")
        if "state" in kwargs:
            kwargs["cursor"]="arrow" if kwargs["state"]=="disabled" else "hand2"
            kwargs["takefocus"]=kwargs["state"]!="disabled"
        super().configure(cnf, **kwargs)
        if hasattr(self, "text"):
            self.draw()

    config = configure

    def change(self, **values):
        self.__dict__.update(values)
        self.draw()

    def release(self, event):
        pressed = self.pressed
        self.pressed = False
        if pressed and 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
            self.invoke()
        self.draw()

    def invoke(self, keyboard=False):
        if self.cget("state") != "disabled" and not self.busy:
            self.keyboard_focus = keyboard
            self.focus_set()
            self.command()
        return "break"

    def set_busy(self, busy):
        self.busy = busy
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        self.animate() if busy else self.draw()

    def animate(self):
        if not self.winfo_exists():
            return
        self.phase = (self.phase+18) % 360
        self.draw()
        if self.busy:
            self.timer = self.after(75, self.animate)

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 3 or h < 3:
            return
        disabled = self.cget("state") == "disabled"
        focused = self.focus_get() == self and self.keyboard_focus
        color, fill, line = MUTED, SURFACE2, BORDER
        if self.variant in ("nav", "preset"):
            fill = self.cget("bg") if self.variant=="nav" else SURFACE
            line = fill if self.variant=="nav" else BORDER
            if self.selected:
                color, fill = TEXT, SURFACE2 if self.variant=="preset" else fill
                line = RED_DIM if self.variant=="preset" else fill
        elif self.variant == "segment":
            fill, line = (RED, RED) if self.selected else (SURFACE, SURFACE)
            color = TEXT if self.selected else DIM
        elif self.variant == "circle":
            fill, line, color = RED, RED, "#ffffff"
        if disabled:
            color, fill, line = FAINT, SURFACE2, BORDER
        elif self.pressed:
            fill = RED_DIM if self.variant in ("circle","segment") and (self.selected or self.variant=="circle") else BORDER2
        elif self.hover:
            color = TEXT
            if self.variant == "circle":
                fill = RED_DIM
            elif self.variant not in ("nav","segment"):
                line, fill = BORDER2, SURFACE2
        if self.busy:
            fill, line, color = RED_DIM, RED_DIM, TEXT
        if self.variant == "circle":
            scale = 4
            image = Image.new("RGBA",(w*scale,h*scale))
            draw = ImageDraw.Draw(image)
            pad = px(5)*scale
            draw.ellipse((pad,pad,w*scale-pad,h*scale-pad),fill=fill)
            if self.busy:
                draw.arc((scale,scale,w*scale-scale,h*scale-scale),self.phase,self.phase+70,
                    fill=RED,width=max(1,px(2)*scale))
            if focused:
                draw.ellipse((scale,scale,w*scale-scale,h*scale-scale),outline=RED_DIM,width=scale)
            self.circle = ImageTk.PhotoImage(image.resize((w,h),Image.Resampling.LANCZOS))
            self.create_image(w/2,h/2,image=self.circle)
        else:
            rounded(self,1,1,w-1,h-1,px(7),fill=fill,outline=line,width=1)
        self.create_text(w/2,h/2,text=("Working…" if self.busy else self.text),fill=color,
            font=("Segoe UI Semibold",10) if self.variant=="circle" else BODY)
        if self.variant=="nav" and self.selected:
            self.create_line(px(14),h-px(2),w-px(14),h-px(2),fill=RED,width=px(2))
        if focused and self.variant!="circle":
            rounded(self,2,2,w-2,h-2,px(6),fill="",outline=RED_DIM,width=px(1))

class Surface(tk.Canvas):
    def __init__(self, parent, height=300):
        super().__init__(parent, height=px(height), bg=BG, bd=0, highlightthickness=0)
        self.body = tk.Frame(self, bg=SURFACE)
        self.window = self.create_window(0,0,anchor="nw",window=self.body)
        self.bind("<Configure>", self.resize)

    def resize(self, event):
        self.delete("border")
        rounded(self,1,1,event.width-1,event.height-1,px(12),fill=SURFACE,outline=BORDER,tags="border")
        self.tag_lower("border")
        margin = px(16)
        self.coords(self.window,margin,margin)
        self.itemconfigure(self.window,width=max(1,event.width-2*margin),height=max(1,event.height-2*margin))

class Preview(tk.Canvas):
    def __init__(self, parent, placeholder, hint="", command=None, source=False):
        super().__init__(parent,bg=SURFACE,highlightthickness=0,bd=0,height=px(200),
            cursor="hand2" if command else "",takefocus=bool(command))
        self.picture = None
        self.placeholder, self.hint = placeholder, hint
        self.command, self.source = command, source
        self.keyboard_focus = True
        self.hover = False
        self.bind("<Configure>",lambda e:self.render())
        self.bind("<Enter>",lambda e:self.hovered(True))
        self.bind("<Leave>",lambda e:self.hovered(False))
        self.bind("<FocusIn>",lambda e:self.render())
        self.bind("<FocusOut>",lambda e:self.focus_left())
        if command:
            self.bind("<Button-1>",self.clicked)
            self.bind("<Return>",lambda e:command())
            self.bind("<space>",lambda e:command())

    def clicked(self,event):
        self.keyboard_focus=False
        self.focus_set()
        self.command()

    def focus_left(self):
        self.keyboard_focus=True
        self.render()

    def hovered(self, value):
        self.hover = value
        self.render()

    def show(self, image):
        self.picture = image
        self.render()

    def render(self):
        self.delete("all")
        w,h = self.winfo_width(),self.winfo_height()
        if min(w,h)<4:
            return
        color = RED_DIM if self.hover and self.command else BORDER2
        if self.picture is not None:
            image = self.picture.copy()
            image.thumbnail((max(1,w-px(16)),max(1,h-px(16))),Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.create_image(w/2,h/2,image=self.photo)
        else:
            rounded(self,1,1,w-1,h-1,px(9),fill=SURFACE,outline=color,
                dash=(px(5),px(4)) if self.source else ())
            cx,cy=w/2,h/2-px(24)
            size=px(12)
            self.create_rectangle(cx-size,cy-size,cx+size,cy+size,outline=DIM,width=px(1))
            self.create_oval(cx-px(7),cy-px(7),cx-px(2),cy-px(2),outline=DIM)
            self.create_line(cx-size,cy+size,cx+px(3),cy-px(2),cx+size,cy+px(7),fill=DIM,width=px(1))
            self.create_text(w/2,h/2+px(12),text=self.placeholder,fill=MUTED,font=BODY,width=max(1,w-px(24)))
            self.create_text(w/2,h/2+px(36),text=self.hint,fill=DIM,font=MONO,width=max(1,w-px(24)))
        if self.focus_get()==self and self.keyboard_focus:
            rounded(self,2,2,w-2,h-2,px(8),fill="",outline=RED_DIM)

class Selectable(tk.Entry):
    def __init__(self,parent,variable,secondary=False):
        super().__init__(parent,textvariable=variable,state="readonly",readonlybackground=SURFACE,
            fg=MUTED if secondary else TEXT,relief="flat",bd=0,highlightthickness=0,
            font=MONO if secondary else ("Segoe UI Semibold",11),width=1,exportselection=False,
            selectbackground=BORDER2,selectforeground=TEXT,cursor="xterm")
        menu=tk.Menu(self,tearoff=False,bg=SURFACE2,fg=TEXT,activebackground=BORDER2,
            activeforeground=TEXT,relief="flat",bd=1)
        menu.add_command(label="Copy",command=lambda:self.copy_value(variable.get()))
        self.bind("<Button-3>",lambda e:menu.tk_popup(e.x_root,e.y_root))
        self.bind("<Control-a>",lambda e:self.select_all())

    def select_all(self):
        self.selection_range(0,"end")
        return "break"

    def copy_value(self,value):
        self.clipboard_clear()
        self.clipboard_append(value)

class Stats(tk.Frame):
    def __init__(self,parent,keys,ratio=False):
        super().__init__(parent,bg=SURFACE)
        self.values={}
        self.fields={}
        self.columnconfigure(tuple(range(len(keys))),weight=1)
        self.ratio=tk.StringVar(value="")
        for index,key in enumerate(keys):
            tk.Label(self,text=key.upper(),bg=SURFACE,fg=DIM,font=("Consolas",9),anchor="w").grid(
                row=0,column=index,sticky="w")
            value=tk.StringVar(value="—")
            self.values[key]=value
            field=Selectable(self,value)
            field.grid(row=1,column=index,sticky="ew",pady=(px(2),0))
            self.fields[key]=field
            if ratio and key.lower()=="bpp":
                self.ratio_field=Selectable(self,self.ratio,secondary=True)
                self.ratio_field.grid(row=2,column=index,sticky="ew")

class Pages(tk.Frame):
    def __init__(self,parent,**kwargs):
        super().__init__(parent,bg=BG,bd=0,highlightthickness=0,**kwargs)
        self.names={}
        self.current=""

    def add(self,page,text):
        self.names[str(page)]=text
        page.place(x=0,y=0,relwidth=1,relheight=1)
        self.select(self.current or page)

    def select(self,page=None):
        if page is None:
            return self.current
        self.current=str(page)
        self.nametowidget(self.current).lift()
        self.event_generate("<<NotebookTabChanged>>",when="tail")

    def tabs(self):
        return tuple(self.names)

    def tab(self,page,option):
        if option!="text":
            raise ValueError(option)
        return self.names[str(page)]
