import gradio as gr
from PBC2_3 import PBC
from PIL import Image
import numpy as np
import ast
import time

SHOW_TRUE_VALUES = False

def update_stroke_count(stroke_mode):
    if stroke_mode == "Auto":
        return gr.update(visible=False), gr.update(value=-1)
    elif stroke_mode == "Manual":
        return gr.update(visible=True, value=40000), gr.update(value=40000)
    
def update_sizeranges(sizerange_mode):
    if sizerange_mode == "Auto":
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=-1), gr.update(value=-1)
    elif sizerange_mode == "Manual":
        return gr.update(visible=True, value=0.3), gr.update(visible=True, value=0.01), gr.update(value=0.3), gr.update(value=0.01)
    
def update_decay(decay_mode):
    if decay_mode == "Auto":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif decay_mode == "Manual":
        return gr.update(visible=True, value=0.5), gr.update(visible=True, value=0.5), gr.update(visible=True, value=0.5)
    
def update_startfrom(startfrom):
    return gr.update(visible=startfrom == "Custom")
    
def display_decay(display_decay_plot, img, stroke_count, sizerange_start, sizerange_end, decay_softness, decay_progress, decay_cutoff):
    if display_decay_plot:
        if img is None:
            max_hw = 512  # Default value if no image is loaded
            min_hw = 512
        else:
            max_hw = max(img.size)
            min_hw = min(img.size)
        if stroke_count == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 20000
            b = 0.0015
            c = 3200
            stroke_count = int(a + b * ((max_hw + c) ** 2))
        if sizerange_start == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.3
            b = 1.00095
            c = 7000
            sizerange_start = a + (1/b)**(stroke_count + c)
        if sizerange_end == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.01
            b = 1.00015
            c = 10200
            sizerange_end = a + (1/b)**(stroke_count + c)

        size_start = int(sizerange_start * (min_hw - 2)) + 2
        size_end = int(sizerange_end * (min_hw - 2)) + 2

        return gr.update(value=PBC.plot_curve_gradio(size_start, size_end, stroke_count, decay_softness, decay_progress, decay_cutoff), y_lim=[0,min_hw], visible=True)
    else:
        return gr.update(visible=False)

def update_effectives_and_plot(val, display_decay_plot, input_image, size_start, size_end, stroke_count, decay_softness, decay_progress, decay_cutoff):
    return gr.update(value=val), display_decay(display_decay_plot, input_image, stroke_count, size_start, size_end, decay_softness, decay_progress, decay_cutoff)

def update_downsample_init(downsample_init):
    return gr.update(visible=downsample_init)

def parameters_selector(parameters):
    if parameters == "Auto":
        # Reset effective parameters to -1 and make all relevant components invisible
        return (
            gr.update(visible=False, value="Auto"),  # Stroke Count Mode
            gr.update(value=-1),      # Effective Stroke Count
            gr.update(visible=False, value="Auto"),  # Size Range Mode
            gr.update(value=-1),      # Effective Size Range Start
            gr.update(value=-1),      # Effective Size Range End
            gr.update(visible=False, value="Auto"),  # Decay Function Mode
            gr.update(value=-1),      # Effective Decay Cutoff
            gr.update(value=-1),      # Effective Decay Softness
            gr.update(value=-1),      # Effective Decay Progress
            gr.update(visible=False, value=True),    # Downsample Init Checkbox
            gr.update(visible=False, value=16),      # Downsample Initial Rate
            gr.update(visible=False, value="Average"),  # Canvas Starting Color
            gr.update(visible=False, value="(255, 0, 0)"),  # Custom Start Colors
            gr.update(visible=False, value="[-10, 0, 5, 20]"),  # Multiplier List
            gr.update(visible=False, value=False),  # Display Decay Function Checkbox
            gr.update(visible=False),  # Multiplier Bitcount Slider
            gr.update(visible=False),  # Multiplier Minimum Slider
            gr.update(visible=False),  # Multiplier Maximum Slider
            gr.update(visible=False),  # Generate Random Multiplier Button
            gr.update(visible=False),  # Generate Uniform Multiplier Button
            gr.update(visible=False),  # Generate Stable Uniform Multiplier Button
            gr.update(visible=False, value="RGB"),  # Color Space Radio
        )
    elif parameters == "Manual":
        # Make all parameters visible without changing their current values
        return (
            gr.update(visible=True),  # Stroke Count Mode
            gr.update(visible=False),  # Effective Stroke Count
            gr.update(visible=True),  # Size Range Mode
            gr.update(visible=False),  # Effective Size Range Start
            gr.update(visible=False),  # Effective Size Range End
            gr.update(visible=True),  # Decay Function Mode
            gr.update(visible=False),  # Effective Decay Cutoff
            gr.update(visible=False),  # Effective Decay Softness
            gr.update(visible=False),  # Effective Decay Progress
            gr.update(visible=True, value=True),    # Downsample Init Checkbox
            gr.update(visible=True),  # Downsample Initial Rate
            gr.update(visible=True, value="Average"),  # Canvas Starting Color
            gr.update(visible=False),  # Custom Start Colors
            gr.update(visible=True),  # Multiplier List
            gr.update(visible=True),  # Display Decay Function Checkbox
            gr.update(visible=True),  # Multiplier Bitcount Slider
            gr.update(visible=True),  # Multiplier Minimum Slider
            gr.update(visible=True),  # Multiplier Maximum Slider
            gr.update(visible=True),  # Generate Random Multiplier Button
            gr.update(visible=True),  # Generate Uniform Multiplier Button
            gr.update(visible=True),  # Generate Stable Uniform Multiplier Button
            gr.update(visible=True),  # Color Space Radio
        )

def mode_selector(mode):
    if mode == "Encode":
        return (
            gr.update(visible=True),   # Image Input
            gr.update(visible=False, value=None),  # PBC File Input
            gr.update(visible=True, value="Auto"),  # Parameters Radio
            gr.update(visible=True, value="Compress", elem_id="compress-button"),  # Compress Button
            gr.update(value=None), # Reset output image
            gr.update(visible=False), # Reset stats box
            gr.update(visible=False), # Reset download button
        )
    elif mode == "Decode":
        return (
            gr.update(visible=False, value=None),  # Image Input
            gr.update(visible=True),   # PBC File Input
            gr.update(visible=False, value="Auto"),  # Parameters Radio
            gr.update(visible=True, value="Decompress", elem_id="decompress-button"),  # Decompress Button
            gr.update(value=None), # Reset output image
            gr.update(visible=False), # Reset stats box
            gr.update(visible=False), # Reset download button
        )

def compress(img_pil, stroke_count, sizerange_start, sizerange_end, decay_softness, decay_progress, decay_cutoff, start_mode, start_custom, mult_list, color_space, downsample_init, downsample_initialize_rate):
    if img_pil is None:
        return Image.fromarray(np.zeros((1,1,3), dtype=np.uint8)), gr.update(value="No image provided. Select an image first before pressing compress.", visible=True), gr.update(visible=False)
    timer = time.time()
    compressed_img, _, bitstream, losses = PBC.compress(
        img_pil = img_pil,
        stroke_count = stroke_count,
        size_range = (sizerange_start, sizerange_end),
        mult_list = ast.literal_eval(mult_list),
        start_mode = start_mode,
        start_custom = ast.literal_eval(start_custom),
        decay_params = {"cutoff": decay_cutoff, "softness": decay_softness, "progress": decay_progress},
        color_space = color_space,
        downsample_initialize = downsample_init,
        downsample_initialize_rate = downsample_initialize_rate
    )
    timer = time.time() - timer

    w, h = img_pil.size
    original_size = w * h * 3
    compressed_size = len(bitstream) / 8

    compressed_size_str = f"Compressed from {(original_size / 1024):.2f} KB to {(compressed_size / 1024):.2f} KB\n"
    compression_rate = f"Compression rate: {(original_size/compressed_size):.2f}x / {(compressed_size/original_size)*100:.2f}%\n"
    losses_string = f"Channel MSE Losses: R: {losses[0]:.0f}, G: {losses[1]:.0f}, B: {losses[2]:.0f}\n"
    average_mse = f"Average MSE Loss: {np.mean(losses):.0f}\n"
    timer = f"Compression time: {timer:.2f} seconds\n"
    return compressed_img, gr.update(value=compressed_size_str + compression_rate + losses_string + average_mse + timer, visible=True), gr.update(value="./compressed.pbc", visible=True)

def decompress(pbc_file):
    if pbc_file is None:
        return Image.fromarray(np.zeros((1,1,3), dtype=np.uint8)), gr.update(value="No file provided. Select a .pbc file first before pressing decompress.", visible=True), gr.update(visible=False)
    if pbc_file.name.split('.')[-1].lower() != 'pbc':
        return Image.fromarray(np.zeros((1,1,3), dtype=np.uint8)), gr.update(value="Invalid file type. Please provide a .pbc file.", visible=True), gr.update(visible=False)
    timer = time.time()
    decompressed_img = PBC.decompress_from_filename(pbc_file)
    timer = time.time() - timer
    
    w, h = decompressed_img.size
    original_size = w * h * 3
    compressed_size = len(open(pbc_file.name, 'rb').read())

    compressed_size_str = f"File was compressed from {(original_size / 1024):.2f} KB to {(compressed_size / 1024):.2f} KB\n"
    compression_rate = f"Compression rate: {(original_size/compressed_size):.2f}x / {(compressed_size/original_size)*100:.2f}%\n"
    timer = f"Decompression time: {timer:.2f} seconds\n"
    return decompressed_img, gr.update(value=compressed_size_str + compression_rate + timer, visible=True), gr.update(visible=False)

def compress_button_script(mode, pbc_file, img_pil, stroke_count, sizerange_start, sizerange_end, decay_softness, decay_progress, decay_cutoff, start_mode, start_custom, mult_list, color_space, downsample_init, downsample_initialize_rate):
    if mode == "Encode":
        return compress(img_pil, stroke_count, sizerange_start, sizerange_end, decay_softness, decay_progress, decay_cutoff, start_mode, start_custom, mult_list, color_space, downsample_init, downsample_initialize_rate)
    elif mode == "Decode":
        return decompress(pbc_file)

css = """
#compress-button {
    width: 40%;
    aspect-ratio: 1;
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

#decompress-button {
    width: 40%;
    aspect-ratio: 1;
    border-radius: 50%;
    background-color: #400000;
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
"""


with gr.Blocks() as demo:
    gr.Markdown(
        """
        ---
        # PBC V2.3
        ### [github.com/EgeEken/PBC](https://github.com/EgeEken/PBC)
        ---
        """
    )

    mode = gr.Radio(choices=["Encode", "Decode"], label="Mode", value="Encode", elem_id="mode-radio")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(label="Upload Image", type="pil")
            pbc_file_input = gr.File(label="Upload .pbc File", visible=False)
            parameters = gr.Radio(choices=["Auto", "Manual"], label="Parameters", value="Auto")

            # STROKE COUNT
            stroke_mode = gr.Radio(choices=["Auto", "Manual"], label="Stroke Count Mode", value="Auto", visible=False)
            stroke_count = gr.Slider(1, 200000, 40000, step=100, label="Stroke Count", visible=False)
            # Intermediate value holders
            effective_stroke_count = gr.Number(value=-1, label="Effective Stroke Count", visible=SHOW_TRUE_VALUES)
            
            # SIZE RANGE
            sizerange_mode = gr.Radio(choices=["Auto", "Manual"], label="Size Range Mode", value="Auto", visible=False)
            sizerange_start = gr.Slider(0.001, 1.0, 0.3, step=0.0001, label="Size Range Start", visible=False)
            sizerange_end = gr.Slider(0.001, 1.0, 0.01, step=0.0001, label="Size Range End", visible=False)
            # Intermediate value holders
            effective_sizerange_start = gr.Number(value=-1, label="Effective Size Range Start", visible=SHOW_TRUE_VALUES)
            effective_sizerange_end = gr.Number(value=-1, label="Effective Size Range End", visible=SHOW_TRUE_VALUES)
            
            # DECAY FUNCTION
            decay_mode = gr.Radio(choices=["Auto", "Manual"], label="Decay Function Mode", value="Auto", visible=False)
            decay_cutoff = gr.Slider(0.0, 1.0, 0.5, label="Cutoff Value", step=0.01, visible=False)
            decay_softness = gr.Slider(0.0, 1.0, 0.5, label="Softness Value", step=0.01, visible=False)
            decay_progress = gr.Slider(0.0, 1.0, 0.5, label="Progress Value", step=0.01, visible=False)
            # Intermediate value holders
            effective_decay_cutoff = gr.Number(value=-1, label="Effective Cutoff Value", visible=SHOW_TRUE_VALUES)
            effective_decay_softness = gr.Number(value=-1, label="Effective Softness Value", visible=SHOW_TRUE_VALUES)
            effective_decay_progress = gr.Number(value=-1, label="Effective Progress Value", visible=SHOW_TRUE_VALUES)
            
            # DECAY PLOT
            display_decay_plot = gr.Checkbox(label="Display Decay Function Plot", value=False, visible=False)
            custom_func_plot = gr.LinePlot(x="x", y="y", visible=False, x_title="Seed", y_title="Size", title="Custom Size Function")
            
            # DOWNSAMPLE INIT RATE
            downsample_init = gr.Checkbox(label="Enable Downsampled Layer Initialization", value=True, visible=False)
            downsample_init_rate = gr.Slider(1, 64, 16, step=0.5, label="Downsample Initial Rate", visible=False)
            
            # STARTING COLORS
            startfrom = gr.Radio(choices=["Average", "Median", "Black", "White", "Custom"], label="Canvas Starting Color", value="Average", visible=False)
            custom_values = gr.Code(value="(255, 0, 0)", lines=1, label="Custom Start Colors (select 'Custom' above)", visible=False)
            
            # MULTIPLIER LIST
            multlist = gr.Code("[-10, 0, 5, 20]", lines=1, language='python', label="Multiplier List", visible=False)

            # generator parameters
            bit_count_slider = gr.Slider(1, 9, 2, step=1, label="Multiplier Bit Count", visible=False)
            mult_min = gr.Slider(-255, 255, -10, step=1, label="Multiplier Minimum", visible=False)
            mult_max = gr.Slider(-255, 255, 20, step=1, label="Multiplier Maximum", visible=False)

            # generator buttons
            randomize_button = gr.Button("Randomize Multiplier List", visible=False)
            uniform_button = gr.Button("Generate Uniform Multiplier List", visible=False)
            stable_button = gr.Button("Generate Stable Uniform Multiplier List", visible=False)

            # COLOR SPACE
            colorspace = gr.Radio(choices=["RGB", "YCbCr"], label="Color Space", value="RGB", visible=False)

        with gr.Column(scale=2):
            gr.HTML("<div style='display: flex; align-items: center; height: 100%; justify-content: center; flex-direction: column;'>")
            compress_button = gr.Button("Compress", elem_id="compress-button")
            gr.HTML("</div>")

        with gr.Column(scale=5):
            output_image = gr.Image(label="Output Image", format="png", sources=None, interactive=False)
            stats = gr.Textbox(label="Bit Statistics", lines=6, max_lines=6, interactive=False, visible=False)
            download_button = gr.DownloadButton(label="Download Compressed .pbc File", value="./compressed.pbc", visible=False)

    compress_button.click(
        compress_button_script, 
        inputs=[mode, pbc_file_input, image_input, effective_stroke_count, effective_sizerange_start, effective_sizerange_end, 
                effective_decay_softness, effective_decay_progress, effective_decay_cutoff, startfrom, 
                custom_values, multlist, colorspace, downsample_init, downsample_init_rate],
        outputs=[output_image, stats, download_button]
    )

    # Mode selector
    mode.change(
        mode_selector,
        inputs=[mode],
        outputs=[
            image_input,         # Image Input
            pbc_file_input,      # PBC File Input
            parameters,          # Parameters Radio
            compress_button,     # Compress Button
            output_image,        # Output Image
            stats,               # Stats Box
            download_button      # Download Button
        ]
    )

    parameters.change(
        parameters_selector,
        inputs=[parameters],
        outputs=[
            stroke_mode,          # Stroke Count Mode
            effective_stroke_count,  # Effective Stroke Count
            sizerange_mode,       # Size Range Mode
            effective_sizerange_start,  # Effective Size Range Start
            effective_sizerange_end,    # Effective Size Range End
            decay_mode,           # Decay Function Mode
            effective_decay_cutoff,  # Effective Decay Cutoff
            effective_decay_softness,  # Effective Decay Softness
            effective_decay_progress,  # Effective Decay Progress
            downsample_init,      # Downsample Init Checkbox
            downsample_init_rate, # Downsample Initial Rate
            startfrom,            # Canvas Starting Color
            custom_values,        # Custom Start Colors
            multlist,             # Multiplier List
            display_decay_plot,   # Display Decay Function Checkbox
            bit_count_slider,     # Multiplier Bitcount Slider
            mult_min,             # Multiplier Minimum Slider
            mult_max,             # Multiplier Maximum Slider
            randomize_button,     # Randomize Multiplier Button
            uniform_button,       # Generate Uniform Multiplier Button
            stable_button,        # Generate Stable Uniform Multiplier Button
            colorspace,           # Color Space Radio
        ]
    )

    stroke_mode.change(
        update_stroke_count,
        inputs=stroke_mode,
        outputs=[stroke_count, effective_stroke_count]
    )

    # update effectives
    stroke_count.change(
        update_effectives_and_plot,
        inputs=[stroke_count, display_decay_plot, image_input, effective_sizerange_start, effective_sizerange_end, stroke_count, effective_decay_softness, effective_decay_progress, effective_decay_cutoff],
        outputs=[effective_stroke_count, custom_func_plot]
    )

    sizerange_mode.change(
        update_sizeranges,
        inputs=sizerange_mode,
        outputs=[sizerange_start, sizerange_end, effective_sizerange_start, effective_sizerange_end]
    )

    # update effectives
    sizerange_end.change(
        update_effectives_and_plot,
        inputs=[sizerange_end, display_decay_plot, image_input, effective_sizerange_start, sizerange_end, effective_stroke_count, effective_decay_softness, effective_decay_progress, effective_decay_cutoff],
        outputs=[effective_sizerange_end, custom_func_plot]
    )

    sizerange_start.change(
        update_effectives_and_plot,
        inputs=[sizerange_start, display_decay_plot, image_input, sizerange_start, effective_sizerange_end, effective_stroke_count, effective_decay_softness, effective_decay_progress, effective_decay_cutoff],
        outputs=[effective_sizerange_start, custom_func_plot]
    )

    # multlist generators
    randomize_button.click(
        lambda bit_count, min_val, max_val: str(PBC.generate_multlist(bit_count, min_val, max_val, "Random")),
        inputs=[bit_count_slider, mult_min, mult_max],
        outputs=[multlist]
    )
    
    uniform_button.click(
        lambda bit_count, min_val, max_val: str(PBC.generate_multlist(bit_count, min_val, max_val, "Uniform")),
        inputs=[bit_count_slider, mult_min, mult_max],
        outputs=[multlist]
    )

    stable_button.click(
        lambda bit_count, min_val, max_val: str(PBC.generate_multlist(bit_count, min_val, max_val, "Stable_Uniform")),
        inputs=[bit_count_slider, mult_min, mult_max],
        outputs=[multlist]
    )

    # startfrom change
    startfrom.change(
        update_startfrom,
        inputs=startfrom,
        outputs=custom_values
    )

    # downsample init click
    downsample_init.change(
        update_downsample_init,
        inputs=downsample_init,
        outputs=downsample_init_rate
    )

    # decay function updates
    decay_mode.change(
        update_decay,
        inputs=decay_mode,
        outputs=[decay_softness, decay_progress, decay_cutoff]
    )

    # update effectives
    decay_softness.change(
        update_effectives_and_plot,
        inputs=[decay_softness, display_decay_plot, image_input, effective_sizerange_start, effective_sizerange_end, effective_stroke_count, decay_softness, effective_decay_progress, effective_decay_cutoff],
        outputs=[effective_decay_softness, custom_func_plot]
    )

    decay_progress.change(
        update_effectives_and_plot,
        inputs=[decay_progress, display_decay_plot, image_input, effective_sizerange_start, effective_sizerange_end, effective_stroke_count, effective_decay_softness, decay_progress, effective_decay_cutoff],
        outputs=[effective_decay_progress, custom_func_plot]
    )

    decay_cutoff.change(
        update_effectives_and_plot,
        inputs=[decay_cutoff, display_decay_plot, image_input, effective_sizerange_start, effective_sizerange_end, effective_stroke_count, effective_decay_softness, effective_decay_progress, decay_cutoff],
        outputs=[effective_decay_cutoff, custom_func_plot]
    )

    # decay plot display
    display_decay_plot.change(
        display_decay,
        inputs=[display_decay_plot, image_input, effective_stroke_count, effective_sizerange_start, effective_sizerange_end, effective_decay_softness, effective_decay_progress, effective_decay_cutoff],
        outputs=[custom_func_plot]
    )
    

if __name__ == "__main__":
    start_time = time.time()
    PBC.preload_numba()
    print(f"Preloading numba took {time.time() - start_time:.2f} seconds")
    demo.launch(share=False, css=css)