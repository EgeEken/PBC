import gradio as gr
import PBC2 as pbc
from PIL import Image
import numpy as np
import ast

def PBC_compressor(img, stroke_count, sr1, sr2, c_o, startfrom, custom_values, ml):
    if sr1 == sr2:
        sr2 += 1
    sizerange = (min(sr1, sr2), max(sr1, sr2))
    
    if startfrom == "Custom":
        sf = ast.literal_eval(custom_values)
    else:
        sf = startfrom
    
    multlist = np.array(ast.literal_eval(ml))
    
    cutoff = c_o / stroke_count
      
    height, width = img.shape[0], img.shape[1]
    compressed, strokes, startcolors = pbc.compress(img=img,
                                                    sizerange=sizerange,
                                                    multlist=multlist,
                                                    stroke_count=stroke_count,
                                                    startfrom=sf,
                                                    cutoff=cutoff)
    
    encoded = pbc.encode_all(height, width, startcolors, sizerange, multlist, strokes, cutoff)
    
    compressed_size = f"Compressed from {(img.nbytes / 1024):.2f} KB to {(len(encoded) / 8 / 1024):.2f} KB\nCompression rate: {(img.nbytes/(len(encoded) / 8)):.2f}x"
    
    return Image.fromarray(compressed), compressed_size, encoded

def update_custom_values_visibility(startfrom):
    return gr.update(visible=startfrom == "Custom")

def update_sizerange_end(sizerange_start, sizerange_end):
    return gr.update(value=max(sizerange_start + 1, sizerange_end))

def generate_random_multlist(bit_count, mi=-255, ma=256):
    if bit_count == 9:
        return str(list(np.random.choice(range(-255, 256), 2 ** bit_count - 1, replace=False)))
    return str(list(np.random.choice(range(mi, ma), 2 ** bit_count, replace=False)))

def generate_uniform_multlist(bit_count, mi=-255, ma=256):
    count = 2 ** bit_count
    if bit_count == 9:
        return str(list(np.arange(-255, 256, 1)))
    return str(list(np.linspace(mi, ma, count, dtype=int)))


def predict_size(multlist, strokecount):
    m_len = len(ast.literal_eval(multlist))
    m_bits = int(np.ceil(np.log2(m_len)))
    header = 88 + m_len*9 + 9
    data = strokecount * m_bits * 4
    return f"Predicted Size: {(header + data) / 8 / 1024:.2f} KB"

def update_predicted_size(multlist, strokecount):
    return gr.update(value=predict_size(multlist, strokecount))

def update_cutoff(strokecount):
    return gr.update(maximum=strokecount)

css = """
#compress-button {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background-color: #4C93FF;
    color: white;
    font-size: 20px;
    font-weight: bold;
    font-family: 'Cascadia', sans-serif;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
}

#predicted {
    font-size: 18px;
    font-weight: bold;
    font-family: 'Cascadia', sans-serif;
    text-align: center;
}
"""


with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        ---
        # PBC Compression V2.1
        ### [github.com/EgeEken/PBC](https://github.com/EgeEken/PBC)
        ---
        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(label="Input Image")
            strokecount = gr.Slider(1, 200000, 40000, step=1000, label="Stroke Count")
            sizerange_start = gr.Slider(2, 200, 8, label="Size Range Start")
            sizerange_end = gr.Slider(3, 400, 200, label="Size Range End")
            cutoff = gr.Slider(0, 40000, 15000, label="Cutoff Value", step=10)
            startfrom = gr.Radio(choices=["Black", "Average", "White", "Custom"], label="Start From", value="Average")
            custom_values = gr.Code(value="(255, 0, 0)", lines=1, label="Custom Start Colors (select 'Custom' above)", visible=False)
            
            
            predicted_size = gr.Markdown("", elem_id="predicted")
            multlist = gr.Code("[-35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]", lines=1, language='python', label="Multiplier List")
            
            bit_count_slider = gr.Slider(1, 9, 4, step=1, label="Multiplier Bit Count")
            mult_min = gr.Slider(-255, 255, -35, step=1, label="Multiplier Minimum")
            mult_max = gr.Slider(-255, 255, 40, step=1, label="Multiplier Maximum")
            randomize_button = gr.Button("Randomize Multiplier List")
            generate_button = gr.Button("Generate Uniform Multiplier List")

        with gr.Column(scale=2):
            gr.HTML("<div style='display: flex; align-items: center; height: 100%; justify-content: center; flex-direction: column;'>")
            compress_button = gr.Button("Compress", elem_id="compress-button")
            gr.HTML("</div>")

        with gr.Column(scale=5):
            output_image = gr.Image(label="Output Image", format="png", sources=None, interactive=False)
            compressed_size = gr.Textbox(label="Compressed Size")
            encoded_text = gr.Textbox(label="Encoded Bits", lines=1, max_lines=5, interactive=False, show_copy_button=True)

    compress_button.click(
        PBC_compressor, 
        inputs=[image_input, strokecount, sizerange_start, sizerange_end, cutoff, startfrom, custom_values, multlist],
        outputs=[output_image, compressed_size, encoded_text]
    )

    startfrom.change(
        update_custom_values_visibility, 
        inputs=startfrom, 
        outputs=custom_values
    )

    sizerange_start.change(
        update_sizerange_end, 
        inputs=[sizerange_start, sizerange_end], 
        outputs=sizerange_end
    )
    
    randomize_button.click(
        generate_random_multlist,
        inputs=[bit_count_slider, mult_min, mult_max],
        outputs=[multlist]
    )
    
    generate_button.click(
        generate_uniform_multlist,
        inputs=[bit_count_slider, mult_min, mult_max],
        outputs=[multlist]
    )
    
    multlist.change(
        update_predicted_size,
        inputs=[multlist, strokecount],
        outputs=[predicted_size]
    )
    
    strokecount.change(
        update_predicted_size,
        inputs=[multlist, strokecount],
        outputs=[predicted_size]
    )
    
    strokecount.change(
        update_cutoff,
        inputs=[strokecount],
        outputs=[cutoff]
    )

if __name__ == "__main__":
    demo.launch(share=False)
