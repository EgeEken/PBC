import gradio as gr
import PBC2 as pbc
from PIL import Image
import numpy as np
import ast

def PBC_compressor(img, stroke_count, sr1, sr2, ml):
    
    if sr1 == sr2:
        sr2 += 1
    sizerange = (min(sr1, sr2), max(sr1, sr2))
    
    multlist = np.array(ast.literal_eval(ml))
    
    height, width = img.shape[0], img.shape[1]
    compressed, strokes, startcolors = pbc.compress(img, sizerange, multlist, stroke_count)
    encoded = pbc.encode_all(height, width, startcolors, sizerange, multlist, strokes)
    
    compressed_size = f"from {(img.nbytes / 1024):.2f} KB to {(len(encoded) / 8 / 1024):.2f} KB\nCompression rate: {(img.nbytes/(len(encoded) / 8)):.2f}x"
    
    return Image.fromarray(compressed), compressed_size, encoded
    
image_input = gr.Image()
strokecount = gr.Slider(1, 100000, 10000, label="Stroke Count")
sizerange_start = gr.Slider(1, 200, 5, label="Size Range Start")
sizerange_end = gr.Slider(1, 200, 30, label="Size Range End")
multlist = gr.Textbox("[-30, 0, 15, 30]", label="Multiplier List")

demo = gr.Interface(fn=PBC_compressor, inputs=[image_input, strokecount, sizerange_start, sizerange_end, multlist], outputs=["image", "text", "text"])


demo.launch(share=True)
