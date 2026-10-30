import ctypes as ct
from ctypes import wintypes as wt
import io
import struct
import time
from PIL import Image, ImageGrab

user = ct.WinDLL("user32", use_last_error=True)
kernel = ct.WinDLL("kernel32", use_last_error=True)
shell = ct.WinDLL("shell32", use_last_error=True)
user.OpenClipboard.argtypes = [wt.HWND]
user.OpenClipboard.restype = wt.BOOL
user.EmptyClipboard.restype = wt.BOOL
user.CloseClipboard.restype = wt.BOOL
user.RegisterClipboardFormatW.argtypes = [wt.LPCWSTR]
user.RegisterClipboardFormatW.restype = wt.UINT
user.SetClipboardData.argtypes = [wt.UINT, ct.c_void_p]
user.SetClipboardData.restype = ct.c_void_p
user.GetClipboardData.argtypes = [wt.UINT]
user.GetClipboardData.restype = ct.c_void_p
user.IsClipboardFormatAvailable.argtypes = [wt.UINT]
user.IsClipboardFormatAvailable.restype = wt.BOOL
kernel.GlobalAlloc.argtypes = [wt.UINT, ct.c_size_t]
kernel.GlobalAlloc.restype = ct.c_void_p
kernel.GlobalLock.argtypes = [ct.c_void_p]
kernel.GlobalLock.restype = ct.c_void_p
kernel.GlobalUnlock.argtypes = [ct.c_void_p]
kernel.GlobalFree.argtypes = [ct.c_void_p]
kernel.GlobalSize.argtypes = [ct.c_void_p]
kernel.GlobalSize.restype = ct.c_size_t
PNG = user.RegisterClipboardFormatW("PNG")
MAX_PIXELS = 32_000_000

def _open(owner):
    for _ in range(5):
        if user.OpenClipboard(owner):
            return
        time.sleep(.02)
    raise OSError("Clipboard is busy. Try again.")

def _put(format_id, data):
    handle = kernel.GlobalAlloc(2, len(data))
    if not handle:
        raise ct.WinError(ct.get_last_error())
    try:
        address = kernel.GlobalLock(handle)
        if not address:
            raise ct.WinError(ct.get_last_error())
        try:
            ct.memmove(address, data, len(data))
        finally:
            kernel.GlobalUnlock(handle)
        if not user.SetClipboardData(format_id, handle):
            raise ct.WinError(ct.get_last_error())
        handle = None
    finally:
        if handle:
            kernel.GlobalFree(handle)

def copy_image(image, owner):
    image = image.convert("RGBA")
    w, h = image.size
    pixels = image.tobytes("raw", "BGRA")
    header = struct.pack("<IiiHHIIiiII", 124, w, -h, 1, 32, 3, len(pixels), 0, 0, 0, 0)
    header += struct.pack("<5I", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, 0x73524742)
    header += bytes(36) + struct.pack("<7I", 0, 0, 0, 4, 0, 0, 0)
    assert len(header) == 124
    png = io.BytesIO()
    image.save(png, "PNG")
    _open(owner)
    try:
        if not user.EmptyClipboard():
            raise ct.WinError(ct.get_last_error())
        _put(PNG, png.getvalue())
        _put(17, header + pixels)
    finally:
        user.CloseClipboard()

def _drop_paths(handle):
    paths=[]
    for index in range(shell.DragQueryFileW(handle,0xFFFFFFFF,None,0)):
        size=shell.DragQueryFileW(handle,index,None,0)
        buffer=ct.create_unicode_buffer(size+1)
        shell.DragQueryFileW(handle,index,buffer,size+1)
        paths.append(buffer.value)
    return paths

def paste():
    if user.IsClipboardFormatAvailable(15):
        _open(None)
        try:
            return _drop_paths(user.GetClipboardData(15))
        finally:
            user.CloseClipboard()
    data = None
    if user.IsClipboardFormatAvailable(PNG):
        _open(None)
        try:
            handle = user.GetClipboardData(PNG)
            size = kernel.GlobalSize(handle) if handle else 0
            if size > 128 * 1024 * 1024:
                raise ValueError("Clipboard image exceeds the supported size.")
            if size:
                address = kernel.GlobalLock(handle)
                if address:
                    try:
                        data = ct.string_at(address, size)
                    finally:
                        kernel.GlobalUnlock(handle)
        finally:
            user.CloseClipboard()
    image = Image.open(io.BytesIO(data)) if data else ImageGrab.grabclipboard()
    if isinstance(image, Image.Image):
        if image.width * image.height > MAX_PIXELS or max(image.size) > 65535:
            raise ValueError("Clipboard image exceeds the supported dimensions.")
        alpha = image.mode in ("RGBA", "LA", "PA") or image.mode == "P" and "transparency" in image.info
        return image.convert("RGBA" if alpha else "RGB")
    return image

WNDPROC = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
user.SetWindowLongPtrW.argtypes = [wt.HWND, ct.c_int, ct.c_void_p]
user.SetWindowLongPtrW.restype = ct.c_void_p
user.CallWindowProcW.argtypes = [ct.c_void_p, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user.CallWindowProcW.restype = ct.c_ssize_t
shell.DragAcceptFiles.argtypes = [wt.HWND, wt.BOOL]
shell.DragQueryFileW.argtypes = [ct.c_void_p, wt.UINT, wt.LPWSTR, wt.UINT]
shell.DragQueryFileW.restype = wt.UINT
shell.DragFinish.argtypes = [ct.c_void_p]

class DropTarget:
    def __init__(self, widget, deliver):
        self.hwnd = widget.winfo_id()
        self.deliver = deliver
        self.alive = True
        self.proc = WNDPROC(self.message)
        self.previous = user.SetWindowLongPtrW(self.hwnd, -4, ct.cast(self.proc, ct.c_void_p))
        if not self.previous:
            raise ct.WinError(ct.get_last_error())
        shell.DragAcceptFiles(self.hwnd, True)

    def message(self, hwnd, message, wparam, lparam):
        if message == 0x233:
            try:
                paths = _drop_paths(wparam)
                self.deliver(paths)
            finally:
                shell.DragFinish(wparam)
            return 0
        if message == 0x82:
            self.close()
        return user.CallWindowProcW(self.previous, hwnd, message, wparam, lparam)

    def close(self):
        if self.alive:
            shell.DragAcceptFiles(self.hwnd, False)
            user.SetWindowLongPtrW(self.hwnd, -4, self.previous)
            self.alive = False
