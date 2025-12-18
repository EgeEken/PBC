import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
from numba import njit, uint64, uint32
import cv2

@njit(inline='always')
def pcg_step(state):
    old_state = state
    state = uint64(old_state * 6364136223846793005 + 1442695040888963407)
    xorshifted = uint32(((old_state >> 18) ^ old_state) >> 27)
    rot = uint32(old_state >> 59)
    out = (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
    return state, out

@njit(fastmath=True)
def get_stroke_coords_rolling(state, r_start, c_start, r_end, c_end):
    # Safety Check
    if r_end <= r_start: r_end = r_start + 1
    if c_end <= c_start: c_end = c_start + 1
    
    range_h = uint32(r_end - r_start)
    range_w = uint32(c_end - c_start)

    state, rnd_row = pcg_step(state)
    state, rnd_col = pcg_step(state)

    row = r_start + int(rnd_row % range_h)
    col = c_start + int(rnd_col % range_w)

    return state, row, col

@njit(fastmath=True)
def process_quadrant_int(height, width, size, q_int, bitcount, padding):
    """ 
    Numba version of process_quadrant_bits.
    Uses integer bitwise ops instead of strings.
    """
    row_start, row_end = 0, height
    col_start, col_end = 0, width
    split_height = True
    
    # Iterate bits from MSB to LSB to match string order "10" -> 1 first, then 0
    for b in range(bitcount - 1, -1, -1):
        bit = (q_int >> b) & 1
        
        if split_height:
            mid = (row_start + row_end) // 2
            if bit == 0:
                row_end = mid
            else:
                row_start = mid
        else:
            mid = (col_start + col_end) // 2
            if bit == 0:
                col_end = mid
            else:
                col_start = mid
        split_height = not split_height
    
    row_end -= size
    col_end -= size

    # Apply padding
    row_start = max(0, row_start - padding)
    col_start = max(0, col_start - padding)
    row_end = min(height - size, row_end + padding)
    col_end = min(width - size, col_end + padding)

    return row_start, col_start, row_end, col_end

@njit(fastmath=True)
def stroke_numba(target_layer, canvas_layer, h, w, row, col, size, mult_arr):
    half = size // 2
    
    # Pre-allocate output array (Fixed size 4, integers)
    # Numba loves fixed arrays.
    stroke_indices = np.zeros(4, dtype=np.int32)
    
    # Unroll the loop manually or use range(4). 
    # Do NOT create a list of slice objects.
    
    for k in range(4):
        # Calculate bounds explicitly using integers
        if k == 0:   # TL
            r_s, r_e = row, row + half
            c_s, c_e = col, col + half
        elif k == 1: # TR
            r_s, r_e = row, row + half
            c_s, c_e = col + half, col + size
        elif k == 2: # BL
            r_s, r_e = row + half, row + size
            c_s, c_e = col, col + half
        elif k == 3: # BR
            r_s, r_e = row + half, row + size
            c_s, c_e = col + half, col + size
            
        # Basic boundary check (Integer math is instant)
        if r_s >= h or c_s >= w: continue
        if r_s >= r_e or c_s >= c_e: continue
        
        # Slicing directly on the array
        t_slice = target_layer[r_s:r_e, c_s:c_e]
        c_slice = canvas_layer[r_s:r_e, c_s:c_e]
        
        if t_slice.size == 0: continue

        # Math Logic
        diff = t_slice - c_slice
        mean_diff = np.mean(diff)
        
        # Argmin manually is often faster than np.argmin for small arrays in Numba, 
        # but np.argmin is acceptable here.
        best_idx = np.argmin(np.abs(mult_arr - mean_diff))
        best_mult = mult_arr[best_idx]
        
        # Update Canvas
        if best_mult != 0:
            # In-place addition with clipping
            # Note: Creating a view 'c_slice' and modifying it modifies the original 'canvas_layer'
            for rr in range(c_slice.shape[0]):
                for cc in range(c_slice.shape[1]):
                    val = c_slice[rr, cc] + best_mult
                    if val > 255: val = 255
                    elif val < 0: val = 0
                    c_slice[rr, cc] = val
        
        stroke_indices[k] = best_idx

    return stroke_indices, canvas_layer

@njit(fastmath=True)
def stroke_numba_decompress(canvas_layer, h, w, row, col, size, mult_arr, stroke_indices):
    half = size // 2
    
    for k in range(4):
        # Calculate bounds explicitly using integers
        if k == 0:   # TL
            r_s, r_e = row, row + half
            c_s, c_e = col, col + half
        elif k == 1: # TR
            r_s, r_e = row, row + half
            c_s, c_e = col + half, col + size
        elif k == 2: # BL
            r_s, r_e = row + half, row + size
            c_s, c_e = col, col + half
        elif k == 3: # BR
            r_s, r_e = row + half, row + size
            c_s, c_e = col + half, col + size
            
        # Basic boundary check (Integer math is instant)
        if r_s >= h or c_s >= w: continue
        if r_s >= r_e or c_s >= c_e: continue
        
        c_slice = canvas_layer[r_s:r_e, c_s:c_e]
        
        best_idx = stroke_indices[k]
        best_mult = mult_arr[best_idx]
        
        # Update Canvas
        if best_mult != 0:
            # In-place addition with clipping
            for rr in range(c_slice.shape[0]):
                for cc in range(c_slice.shape[1]):
                    val = c_slice[rr, cc] + best_mult
                    if val > 255: val = 255
                    elif val < 0: val = 0
                    c_slice[rr, cc] = val

    return canvas_layer

class PBC:
    """# Probabilistic Brush Compression (PBC) \n
    ---
    ### Developed by **Ege Eken** (https://github.com/EgeEken/PBC) \n
    Current Version: **V2.3** (2025) \n\n
    ---
    This is a lossy image compression algorithm that compresses images into a series of brush stroke instructions.\n
    For more information, visit the [GitHub Repository](https://github.com/EgeEken/PBC)
    """

    @classmethod
    def preload_numba(cls):
        """ Dummy compression simulation to preload Numba compiled functions."""
        dummy_img = Image.fromarray(np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8))
        cls.compress(dummy_img, stroke_count=10, quadrant_warmup_time=0.0)


    @staticmethod
    def rgb_to_ycbcr(img):
        xform = np.array([[0.299, 0.587, 0.114],
                        [-0.168736, -0.331264, 0.5],
                        [0.5, -0.418688, -0.081312]])
        ycbcr = img.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.clip(ycbcr, 0, 255).astype(np.uint8)

    @staticmethod
    def ycbcr_to_rgb(img):
        xform = np.array([[1, 0, 1.402],
                        [1, -0.344136, -0.714136],
                        [1, 1.772, 0]])
        rgb = img.astype(float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    @classmethod
    def get_decay_curve(cls, length, start, end, cutoff, softness, progress, display_autos=False):
        """Pre-calculates the brush size for every stroke index.
        - cutoff between 0 and 3, 3 chosen arbitrarily (but reasonably) to represent no cutoff, so there will be a jump at 2.9 - 3
        - softness between 0 and 1, 0 being instant jump, 1 being a smooth sigmoid curve
        - progress between 0 and 1, 0 being fully linear, 1 being fully sigmoid
        - values -1 for cutoff, softness, progress represent default behavior (cutoff dependent on length, softness 0.5, progress 0.5) \n
        Returns an array of brush sizes for each stroke index.
        """
        cutoff_flag = False
        softness_flag = False
        progress_flag = False
        if cutoff == -1:
            cutoff_flag = True
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.01
            b = 1.0000115
            c = 15000
            cutoff = a + (1/b)**(length + c)
            if display_autos:
                print(f'Auto-calculated cutoff: {cutoff:.4f}')
        if softness == -1:
            softness_flag = True
            softness = 0.5
            if display_autos:
                print(f'Auto-calculated softness: {softness:.4f} (0.5 by default)')
        if progress == -1:
            progress_flag = True
            progress = 0.5
            if display_autos:
                print(f'Auto-calculated progress: {progress:.4f} (0.5 by default)')

        cutoff = round(cutoff, 4)
        softness = round(softness, 4)
        progress = round(progress, 4)

        decay_bitstream = ""
        # cutoff
        decay_bitstream += "1" if cutoff_flag else "0"
        decay_bitstream += cls.encode_float(cutoff, decimals=4, bitcount=20)
        # softness
        decay_bitstream += "1" if softness_flag else "0"
        decay_bitstream += cls.encode_float(softness, decimals=4, bitcount=20)
        # progress
        decay_bitstream += "1" if progress_flag else "0"
        decay_bitstream += cls.encode_float(progress, decimals=4, bitcount=20)


        x = np.linspace(0, length, length)
    
        if cutoff <= 0:
            return np.full(length, end, dtype=int), decay_bitstream
        lencut = length * cutoff
        if lencut >= length * 3:
            return np.full(length, start, dtype=int), decay_bitstream

        # Linear Component
        lin = start + (x / (length * cutoff)) * (end - start)
        
        # Sigmoid Component
        if softness <= 0: 
            sig = np.where(x >= lencut / 2, end, start)
        else:
            k = 1.0 / softness * (np.sqrt(abs(end - start)) / length)
            sig = start + (end - start) / (1 + np.exp(-k * (x - lencut / 2)))

        curve = progress * sig + (1 - progress) * lin
        mask = x >= lencut
        curve[mask] = end
        return (curve // 2 * 2).astype(int), decay_bitstream
    
    @staticmethod
    def process_quadrant_bits(height, width, size, quadrant_bits, quadrant_padding=0):
        # for each bit, alternate between vertical and horizontal split, for example
        # "0", length 1 means vertical split, 0 is left half, 1 is right half
        # "10", length 2 means vertical first, then horizontal split, 1 is right half, then 0 is top half giving top-right quadrant
        # "111", length 3 means vertical, horizontal, vertical splits, 1 is right half, then 1 is bottom half, then 1 is right half again giving right quadrant of bottom-right quadrant
        split_height = True 
        
        row_start, row_end = 0, height
        col_start, col_end = 0, width
        
        for bit in quadrant_bits:
            if split_height:
                mid = (row_start + row_end) // 2
                if bit == '0':
                    row_end = mid
                else:
                    row_start = mid
            else:
                mid = (col_start + col_end) // 2
                if bit == '0':
                    col_end = mid
                else:
                    col_start = mid
            split_height = not split_height
        
        row_end -= size
        col_end -= size

        # Apply padding
        row_start = max(0, row_start - quadrant_padding)
        col_start = max(0, col_start - quadrant_padding)
        row_end = min(height - size, row_end + quadrant_padding)
        col_end = min(width - size, col_end + quadrant_padding)

        return row_start, col_start, row_end, col_end
    
    @staticmethod
    def get_quadrant_bitcount(height, width, size, max_bitcount = 8):
        """Calculates the maximum number of quadrant bits (divisions) that can be used given image dimensions and brush size."""
        curr_height, curr_width = height, width
        bitcount = 0
        split_height = True
        
        while bitcount < max_bitcount:
            if split_height:
                if curr_height // 2 < size:
                    break
                curr_height //= 2
            else:
                if curr_width // 2 < size:
                    break
                curr_width //= 2
            split_height = not split_height
            bitcount += 1
        return bitcount

    @classmethod
    def select_quadrant(cls, error_layer, quadrant_bitcount, selection_criteria="Sum"):
        """Selects the quadrant with the most error between target and canvas images."""
        if quadrant_bitcount == 0:
            return ""
        height, width = error_layer.shape
        max_error = -1
        best_quadrant = ""
        
        fmt_str = f'0{quadrant_bitcount}b'
        
        for i in range(2 ** quadrant_bitcount):
            bits = format(i, fmt_str)
            r_s, c_s, r_e, c_e = cls.process_quadrant_bits(height, width, 0, bits) # 0 since we want the full area
            
            region = error_layer[r_s:r_e, c_s:c_e]
            if region.size == 0: continue
            
            if selection_criteria == "Max":
                error_quad = np.max(region)
            elif selection_criteria == "Min":
                error_quad = np.min(region)
            elif selection_criteria == "Sum":
                error_quad = np.sum(region)
            else: # default to Sum
                print(f'Warning: Unknown selection_criteria "{selection_criteria}", defaulting to "Sum".')
                error_quad = np.sum(region)
            if error_quad > max_error:
                max_error = error_quad
                best_quadrant = bits
        return best_quadrant
    
    @staticmethod
    def get_quadrant_size(height, width, quadrant_bitcount):
        """FOR DEBUGGING/INFO PURPOSES, NOT USED IN ALGORITHM
        Calculates the size of the quadrant given image dimensions and quadrant bits."""
        curr_height, curr_width = height, width
        vertical = True
        for _ in range(quadrant_bitcount):
            if vertical:
                curr_height //= 2
            else:
                curr_width //= 2
            vertical = not vertical
        return curr_height, curr_width

    @classmethod
    def get_stroke_params(cls, seed, height, width, size, quadrant_input=None, quadrant_padding=0):
        """Deterministic random generation for brush position."""
        if quadrant_input is None:
            r_start, c_start = 0, 0
            r_end, c_end = height - size, width - size
        else:
            r_start, c_start, r_end, c_end = cls.process_quadrant_bits(height, width, size, quadrant_input, quadrant_padding=quadrant_padding)

        rng = np.random.RandomState(seed) 
        
        # ensure high > low 
        r_end = max(r_end, r_start + 1)
        c_end = max(c_end, c_start + 1)

        row = rng.randint(r_start, r_end)
        col = rng.randint(c_start, c_end)
        
        return row, col
    
    @staticmethod
    def generate_multlist(bit_count, min_val, max_val, mode="Stable_Uniform"):
        count = 2 ** bit_count
        if min_val > max_val:
            print("Warning: min_val greater than max_val, swapping values.")
            min_val, max_val = max_val, min_val

        if min_val + 1 >= max_val:
            print("Warning: range(min, max) is one element, returning single value list.")
            return [min_val]

        if max_val - min_val < count:
            print("Warning: Range smaller than count, returning full range, might not require the given bit count.")
            return list(range(min_val, max_val))

        if mode == "Random":
            vals = np.random.choice(range(min_val, max_val), count, replace=False)
            vals = sorted(vals.tolist()) 
            return vals
        
        elif mode == "Uniform":
            vals = np.linspace(min_val, max_val, count, dtype=int)
            return vals.tolist()
        
        elif mode == "Stable_Uniform":
            # ensures there is at least one 0 for stability, or a value within -1 to 1 range
            # prioritizes replacing the larger magnitude values rather than the smaller ones
            vals = np.linspace(min_val, max_val, count, dtype=int).tolist()
            closest_to_zero = min(vals, key=lambda x: abs(x))
            if abs(closest_to_zero) > 1:
                # no value close to zero, replace the largest magnitude value with 0
                vals.remove(max(vals, key=lambda x: abs(x)))
                vals.append(0)
            return sorted(vals)
    
    @staticmethod
    def channel_cycle_strategy(full_error_layer, strategy="Smart", selection_criteria="Min"):
        """Channel cycling strategy, returns a list of 3 channel indices based on error prioritization.

        Strategies:
        - "Strict", [x, x, x] where x is the channel with highest error
        - "Balanced", [x, x, y] where x is highest error channel, y is second highest
        - "Default", [0, 1, 2] standard cycling
        - "Smart", uses either strict, balanced, or default based on how much higher the error is

        Selection Criteria:
        - "Sum", uses sum of absolute errors
        - "Max", uses maximum absolute error
        - "Min", uses minimum absolute error
        """

        if strategy == "Default":
            return [0, 1, 2]
        
        channel_errors = []
        for ch in range(3):
            layer = full_error_layer[:, :, ch]
            if selection_criteria == "Sum":
                channel_errors.append(np.sum(layer))
            elif selection_criteria == "Max":
                channel_errors.append(np.max(layer))
            elif selection_criteria == "Min":
                channel_errors.append(np.min(layer))
            elif selection_criteria == "Median":
                channel_errors.append(np.median(layer))
            else:
                print(f'Warning: Unknown selection_criteria "{selection_criteria}", defaulting to "Sum".')
                channel_errors.append(np.sum(layer))
        sorted_channels = sorted(range(3), key=lambda x: channel_errors[x], reverse=True)

        if strategy == "Strict":
            res =  [sorted_channels[0]] * 3
        elif strategy == "Balanced":
            res = [sorted_channels[0], sorted_channels[0], sorted_channels[1]]
        elif strategy == "Smart":
            if channel_errors[sorted_channels[0]] > 2 * channel_errors[sorted_channels[1]]:
                res = [sorted_channels[0]] * 3
            elif channel_errors[sorted_channels[1]] > 2 * channel_errors[sorted_channels[2]]:
                res = [sorted_channels[0], sorted_channels[0], sorted_channels[1]]
            else:
                res = [0, 1, 2]
        else:
            res = [0, 1, 2]
        
        return res

    @classmethod
    def plot_curve_gradio(cls, sr_start, sr_end, stroke_count, softness, progress, cutoff, sample_count=100):
        s_min, s_max = min(sr_start, sr_end), max(sr_start, sr_end)
        x = np.linspace(0, stroke_count - 1, sample_count, dtype=int)
        y, _ = cls.get_decay_curve(stroke_count, s_max, s_min, cutoff, softness, progress)
        y_sampled = y[x]
        return pd.DataFrame({"x": x, "y": y_sampled})
    
    @classmethod
    def plot_curve_plt(cls, sr_start, sr_end, stroke_count, softness, progress, cutoff, q_warmup=None, cycle_warmup=None):
        s_min, s_max = min(sr_start, sr_end), max(sr_start, sr_end)
        x = range(stroke_count)
        y, _ = cls.get_decay_curve(stroke_count, s_max, s_min, cutoff, softness, progress)
        plt.plot(x, y)
        # if theres warmups, add vertical lines for them, labeled
        if q_warmup is not None:
            plt.axvline(x=q_warmup, color='purple', linestyle='--', label=f'Quadrant Warmup = {q_warmup/stroke_count:.2f}')
            plt.text(q_warmup, s_max*0.9, 'Quadrant Warmup', color='purple', verticalalignment='bottom')
        if cycle_warmup is not None:
            plt.axvline(x=cycle_warmup, color='orange', linestyle='--', label=f'Channel Cycle Warmup = {cycle_warmup/stroke_count:.2f}')
            plt.text(cycle_warmup, s_max*0.8, 'Channel Cycle Warmup', color='orange', verticalalignment='bottom')
        plt.xlabel("Stroke Index")
        plt.ylabel("Brush Size")
        plt.ylim(0, s_max * 1.1)
        plt.title("Brush Size Decay Curve")
        plt.legend()
        plt.grid()
        plt.show()
    
    @staticmethod
    def encode_int(int_num, signed=True, bitcount=16):
        """ Encodes an integer into its binary representation with given bitcount. """
        int_num = int(int_num)
        if signed:
            bitcount -= 1
            if int_num < 0:
                return "0" + format(-int_num, f'0{bitcount}b')
            else:
                return "1" + format(int_num, f'0{bitcount}b')
        return format(int_num, f'0{bitcount}b')
    
    @classmethod
    def encode_float(cls, float_num, decimals=4, bitcount=20):
        """ Encodes a float between 0 and 1 into an unsigned integer binary representation with given bitcount. """
        int_repr = float_num * (10 ** decimals)
        return cls.encode_int(int(int_repr), signed=False, bitcount=bitcount)

    @staticmethod
    def decode_int(bitstream, signed=True):
        """ Decodes an integer from its binary representation. """
        if signed:
            sign_bit = bitstream[0]
            int_value = int(bitstream[1:], 2)
            return -int_value if sign_bit == '0' else int_value
        else:
            return int(bitstream, 2)
            
    @classmethod
    def decode_float(cls, bitstream, decimals=4):
        """ Decodes a float from its binary representation. """
        int_repr = cls.decode_int(bitstream, signed=False)
        return int_repr / (10 ** decimals)

    @staticmethod
    def downsample_image(img_pil, downsample_rate, downsample_alg=Image.BICUBIC):
        """Downsamples the image by the given rate."""
        if downsample_rate <= 1:
            return img_pil
        new_width = img_pil.width // downsample_rate
        new_height = img_pil.height // downsample_rate
        return img_pil.resize((int(new_width), int(new_height)), downsample_alg)

    @staticmethod
    def upsample_image(img_pil, original_size, downsample_alg=Image.BICUBIC):
        """Upsamples the image to the original size."""
        return img_pil.resize(original_size, downsample_alg)
    
    @staticmethod
    def _bits_to_bytes(bitstring):
        """Converts a string of '0'/'1' to a bytearray."""
        # Pad with zeros to make length multiple of 8
        original_len = len(bitstring)
        pad_len = (8 - (original_len % 8)) % 8
        bitstring += '0' * pad_len
        
        byte_data = bytearray()
        for i in range(0, len(bitstring), 8):
            byte = bitstring[i:i+8]
            byte_data.append(int(byte, 2))
            
        return byte_data, pad_len

    @staticmethod
    def _bytes_to_bits(byte_data, pad_len):
        """Converts bytes back to string of '0'/'1', removing padding."""
        bitstring = ""
        for byte in byte_data:
            bitstring += format(byte, '08b')
        
        if pad_len > 0:
            bitstring = bitstring[:-pad_len]
        return bitstring

    @staticmethod
    def array_to_bitstream(array):
        """Converts a numpy array to a bitstream string."""
        flat_array = array.flatten()
        bitstream = ''.join(format(val, '08b') for val in flat_array)
        return bitstream
    
    @staticmethod
    def bitstream_to_array(bitstream, h, w, channels=3):
        """Converts a bitstream string back to a numpy array of given shape."""
        total_values = h * w * channels
        array = np.zeros(total_values, dtype=np.uint8)
        
        for i in range(total_values):
            byte_str = bitstream[i*8:(i+1)*8]
            array[i] = int(byte_str, 2)
        
        return array.reshape((h, w, channels))



    @classmethod
    def compress(cls, img_pil, stroke_count=-1, size_range=(-1, -1), mult_list=[-10, 0, 5, 20], start_mode="Average",
                 start_custom=(128, 128, 128), decay_params={'cutoff': -1, 'softness': -1, 'progress': -1},
                 strokes_per_quadrant=100, quadrant_warmup_time=-1, quadrant_max_bits=8, quadrant_padding=4, quadrant_selection_criteria="Sum",
                 channel_cycle="Smart", strokes_per_channel_cycle=100, channel_cycle_warmup_time=0.9, cycle_selection_criteria="Min",
                 color_space="RGB", downsample_rate=-1, display_autos=False,
                 save_filename=-1, use_numba=True, downsample_initialize=True, downsample_initialize_rate=16, downsample_alg=Image.BICUBIC):

        if img_pil is None:
            return None, "Please upload an image first."
        
        if save_filename == -1:
            save_filename = f"compressed.pbc"

        if color_space == "YCbCr":
            img_pil = Image.fromarray(np.array(img_pil.convert("YCbCr")))
        

        bitstream = ""
        original_size = img_pil.size
        ori_w, ori_h = original_size

        if downsample_rate == -1:
            if min(original_size) < 600:
                downsample_rate = 1
            else:
                downsample_rate = min(original_size) / 500
                if display_autos:
                    print(f'Auto-calculated downsample_rate: {downsample_rate}')

        if downsample_rate > 1:
            img_pil_downsampled = cls.downsample_image(img_pil, downsample_rate, downsample_alg=downsample_alg)
            img = np.array(img_pil_downsampled, dtype=np.int16)
            # (HEADER BITS) downsample flag (1=downsampled)
            bitstream += "1"
            # original width and height bits (16 bits each)
            bitstream += cls.encode_int(ori_w, signed=False, bitcount=16)
            bitstream += cls.encode_int(ori_h, signed=False, bitcount=16)
        else:
            img = np.array(img_pil, dtype=np.int16)
            # (HEADER BITS) downsample flag (0=not downsampled)
            bitstream += "0"

        if color_space == "YCbCr":
            # (HEADER BITS) color space bit YCbCr=1
            bitstream += "1"
        else:
            # (HEADER BITS) color space bit RGB=0
            bitstream += "0"

        h, w = img.shape[:2]

        if stroke_count == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 20000
            b = 0.0015
            c = 3200
            stroke_count = int(a + b * ((max(original_size) + c) ** 2))
            if display_autos:
                print(f'Auto-calculated stroke_count: {stroke_count} (Based on {original_size[0]}x{original_size[1]})')

        # (HEADER BITS) image size bits (max image height and width allowed is 65535 (16 bits each), should be enough for any reasonable use case)
        bitstream += cls.encode_int(h, signed=False, bitcount=16) # unsigned, obviously
        bitstream += cls.encode_int(w, signed=False, bitcount=16) # unsigned, obviously
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        # (HEADER BITS) stroke count bits (stroke counts above 1M probably unfeasible, 20 bits should be enough)
        bitstream += cls.encode_int(stroke_count, signed=False, bitcount=20) # unsigned, obviously
        
        if start_mode == "Black": start_color = (0, 0, 0) if color_space == "RGB" else (0, 128, 128)
        elif start_mode == "White": start_color = (255, 255, 255) if color_space == "RGB" else (255, 128, 128)
        elif start_mode == "Custom": start_color = start_custom
        elif start_mode == "Average" or start_mode == "Mean": start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
        elif start_mode == "Median": start_color = (int(np.median(r)), int(np.median(g)), int(np.median(b)))
        elif start_mode == "True Median": start_color = np.median(img.reshape(-1, 3), axis=0).astype(int)
        elif start_mode == "Random": start_color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
        else:
            print(f'Warning: Unknown start_mode "{start_mode}", defaulting to "Average".')
            start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))

        if display_autos:
            print(f'Start color selected: R={start_color[0]}, G={start_color[1]}, B={start_color[2]}')

        # (HEADER BITS) start color bits (0-255 for each channel, 8 bits each)
        bitstream += cls.encode_int(start_color[0], signed=False, bitcount=8) # R
        bitstream += cls.encode_int(start_color[1], signed=False, bitcount=8) # G
        bitstream += cls.encode_int(start_color[2], signed=False, bitcount=8) # B


        if downsample_initialize:
            if downsample_initialize_rate < 32:
                if stroke_count > 20000:
                    if decay_params["cutoff"] == -1:
                        decay_params["cutoff"] = 0.3
                    if size_range == (-1, -1):
                        size_range = (0.05, 0.01)
                    if quadrant_warmup_time == -1:
                        quadrant_warmup_time = 0.1
                else:
                    if decay_params["cutoff"] == -1:
                        decay_params["cutoff"] = 0.7
                    if size_range == (-1, -1):
                        size_range = (0.1, 0.03)
                    if quadrant_warmup_time == -1:
                        quadrant_warmup_time = 0.7

        if downsample_initialize:
            bitstream += "1" # downsample initialize flag bit
            n_h, n_w = int(h/downsample_initialize_rate), int(w/downsample_initialize_rate)
            bitstream += cls.encode_int(n_h, signed=False, bitcount=10) # downsampled height bits
            bitstream += cls.encode_int(n_w, signed=False, bitcount=10) # downsampled width bits
            if downsample_rate > 1:
                canvas = np.array(img_pil_downsampled.resize((n_w, n_h), downsample_alg), dtype=np.uint8)
            else:
                canvas = np.array(img_pil.resize((n_w, n_h), downsample_alg), dtype=np.uint8)
            #bitstream += "1" * (n_h * n_w * 3 * 8)
            if color_space == "YCbCr":
                canvas = cls.ycbcr_to_rgb(canvas)
            bitstream += cls.array_to_bitstream(canvas)
            canvas = np.array(Image.fromarray(canvas).resize((w, h), downsample_alg), dtype=np.int16)
            if color_space == "YCbCr":
                canvas = cls.rgb_to_ycbcr(canvas).astype(np.int16)
            if display_autos:
                print(f'Downsample initialize test enabled with rate {downsample_initialize_rate}, canvas initialized from downsampled image.')
        else:
            canvas = np.full((h, w, 3), start_color, dtype=np.int16)
            bitstream += "0" # downsample initialize flag bit

        size_start = size_range[0]
        size_end = size_range[1]
        if size_start == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.3
            b = 1.00095
            c = 7000
            size_start = a + (1/b)**(stroke_count + c)
            if display_autos:
                print(f"Auto-calculated size start: {size_start:.3f}")
        if size_end == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.01
            b = 1.00015
            c = 10200
            size_end = a + (1/b)**(stroke_count + c)
            if display_autos:
                print(f"Auto-calculated size end: {size_end:.3f}")

        size_start = int(size_start * (min(h, w) - 2)) + 2
        size_end = int(size_end * (min(h, w) - 2)) + 2

        if display_autos:
            print(f"Calculated size range: {size_start} to {size_end} pixels.")

        # maximum and minimum sizes are the same bitcount as the height and width, since size cannot be larger than image dimensions
        # (HEADER BITS) size start and end bits
        bitstream += cls.encode_int(size_start, signed=False, bitcount=16)
        bitstream += cls.encode_int(size_end, signed=False, bitcount=16)

        sizes, decay_params_encoding = cls.get_decay_curve(stroke_count, size_start, size_end,
                                    decay_params['cutoff'], decay_params['softness'], decay_params['progress'], display_autos=display_autos)
        
        # (HEADER BITS) decay parameters bits
        bitstream += decay_params_encoding

        encoded_strokes = []
        mult_arr = np.array(mult_list)
        multlist_len = len(mult_arr)
        multlist_bitcount = int(np.ceil(np.log2(multlist_len)))
        # (HEADER BITS) multlist length bits
        # since we need to know how many multipliers there are to decode them later, there's no ending delimiter
        bitstream += cls.encode_int(multlist_len, signed=False, bitcount=9) # max amount of multipliers possible: (-255, 255) so 512 multipliers, so 9 bits for the number
        # (HEADER BITS) multlist bits
        for mult in mult_arr:
            bitstream += cls.encode_int(mult, signed=True, bitcount=9) # signed, since multipliers can be negative, 9 bits cover the -255 to 255 range

        if quadrant_warmup_time == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.75
            b = 0.000014
            c = -1000
            d = 550000
            quadrant_warmup_time = a - (b*(stroke_count + c) ** 2) / d
            quadrant_warmup_time = max(0.0, min(quadrant_warmup_time, 1.0))
            if display_autos:
                print(f'Auto-calculated quadrant_warmup_time: {quadrant_warmup_time:.4f}')


        quadrant_warmup_time = int(quadrant_warmup_time * stroke_count)
        if display_autos:
            print(f'Calculated quadrant_warmup_time: {quadrant_warmup_time} strokes.')

        # (HEADER BITS) quadrant warmup time bits
        bitstream += cls.encode_int(quadrant_warmup_time, signed=False, bitcount=20)
        # (HEADER BITS) strokes per quadrant bits
        bitstream += cls.encode_int(strokes_per_quadrant, signed=False, bitcount=20)
        # (HEADER BITS) quadrant max bits bits
        bitstream += cls.encode_int(quadrant_max_bits, signed=False, bitcount=8) # max 255 bits should be way more than enough, default is 8
        # (HEADER BITS) quadrant padding bits
        bitstream += cls.encode_int(quadrant_padding, signed=False, bitcount=8) # max 255 pixel padding should be way more than enough, default is 4

        quadrant_switch_counters = [quadrant_warmup_time//3, quadrant_warmup_time//3, quadrant_warmup_time//3]
        quadrant_inputs = [None, None, None]

        if not channel_cycle:
            channel_cycle = None
        channel_selector = [0, 1, 2] # R=0, G=1, B=2
        channel_cycle_timer = int(channel_cycle_warmup_time * stroke_count)
        # by default its 0 1 2 to cycle through 3 channels
        # but it can be modified dynamically to prioritize certain channels with more error
        # for example [0, 0, 1] would be that it does R R G R R G ...

        # (HEADER BITS) channel cycle bool bit
        bitstream += "1" if channel_cycle else "0"
        if channel_cycle:
            # (HEADER BITS) strokes per channel cycle bits
            bitstream += cls.encode_int(strokes_per_channel_cycle, signed=False, bitcount=20)
            # (HEADER BITS) channel cycle warmup time bits
            bitstream += cls.encode_int(int(channel_cycle_warmup_time * stroke_count), signed=False, bitcount=20)

        header_bits = len(bitstream)
        # print(f"Header bits: {header_bits} bits")
        # (MAIN LOOP)

        rng_state = None
        if use_numba:
            # Initialize Numba RNG state
            rng_state = np.uint64(2003)

        for i in range(stroke_count):
            channel_idx = channel_selector[i % 3]
            target_layer = img[:, :, channel_idx]
            canvas_layer = canvas[:, :, channel_idx]
            size = sizes[i]

            if quadrant_switch_counters[channel_idx] <= 0:
                error_layer = np.abs(target_layer - canvas_layer)
                quadrant_bitcount = cls.get_quadrant_bitcount(h, w, size, quadrant_max_bits)
                quadrant_inputs[channel_idx] = cls.select_quadrant(error_layer, quadrant_bitcount, selection_criteria=quadrant_selection_criteria)
                # (MAIN LOOP BITS) quadrant bits
                bitstream += quadrant_inputs[channel_idx]
                quadrant_switch_counters[channel_idx] = strokes_per_quadrant
            if channel_cycle:
                if channel_cycle_timer <= 0:
                    # Re-evaluate channel priorities based on current error
                    full_error_layer = np.abs(img - canvas)
                    channel_selector = cls.channel_cycle_strategy(full_error_layer, strategy=channel_cycle, selection_criteria=cycle_selection_criteria)
                    # (MAIN LOOP BITS) channel cycle bits
                    for ch in channel_selector:
                        bitstream += cls.encode_int(ch, signed=False, bitcount=2) # 2 bits to represent 0-2
                    channel_cycle_timer = strokes_per_channel_cycle

            # RETURNS ROW, COL
            if use_numba:
                # 1. PRE-CALCULATE BOUNDARIES IN PYTHON
                # This avoids passing Strings/None to Numba
                if quadrant_inputs[channel_idx] is None:
                    r_s, c_s = 0, 0
                    r_e, c_e = h - size, w - size
                else:
                    q_int = cls.decode_int(quadrant_inputs[channel_idx], signed=False)
                    q_bitcount = len(quadrant_inputs[channel_idx])
                    r_s, c_s, r_e, c_e = process_quadrant_int(h, w, size, q_int, q_bitcount, quadrant_padding)
                rng_state, row, col = get_stroke_coords_rolling(rng_state, r_s, c_s, r_e, c_e)
                rng_state = np.uint64(rng_state)
            else:
                row, col = cls.get_stroke_params(i, h, w, size, quadrant_input=quadrant_inputs[channel_idx], quadrant_padding=quadrant_padding)

            # APPLIES STROKE (WITH ITS 4 SLICES)
            if use_numba:
                stroke_indices, canvas_layer = stroke_numba(target_layer, canvas_layer, h, w, row, col, size, mult_arr)
            else:
                half = size // 2
                # Slices are [Row, Col]
                slices = [
                    (slice(row, row+half), slice(col, col+half)),
                    (slice(row, row+half), slice(col+half, col+size)),
                    (slice(row+half, row+size), slice(col, col+half)),
                    (slice(row+half, row+size), slice(col+half, col+size))
                ]
                
                stroke_indices = []
                for sl in slices:
                    # Basic boundary check (Never supposed to happen, but just in case)
                    if sl[0].start >= h or sl[1].start >= w:
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nSlice start out of bounds: {sl}, image size: ({h}, {w})\n----------\n\n\n")
                        continue

                    if sl[0].start >= sl[0].stop or sl[1].start >= sl[1].stop:
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nInvalid slice with start >= stop: {sl}\n----------\n\n\n")
                        continue
                    
                    # Extract region
                    target_slice = target_layer[sl]
                    canvas_slice = canvas_layer[sl]

                    # Another safety check for empty slices (which should not happen)
                    if target_slice.size == 0: 
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nEmpty slice encountered: {sl}\n----------\n\n\n")
                        continue

                    diff = target_slice - canvas_slice
                    mean_diff = np.mean(diff)
                    
                    best_idx = np.argmin(np.abs(mult_arr - mean_diff))
                    best_mult = mult_arr[best_idx]
                    
                    canvas_layer[sl] = np.clip(canvas_layer[sl] + best_mult, 0, 255)
                    stroke_indices.append(best_idx)

            # (MAIN LOOP BITS) stroke indices bits
            for idx in stroke_indices:
                bitstream += cls.encode_int(idx, signed=False, bitcount=multlist_bitcount)

            encoded_strokes.append(stroke_indices)
            quadrant_switch_counters[channel_idx] -= 1
            if channel_cycle:
                channel_cycle_timer -= 1


        canvas = np.clip(canvas, 0, 255)

        if color_space == "YCbCr":
            final_img = cls.ycbcr_to_rgb(canvas).astype(np.uint8)
            img_pil = Image.fromarray(cls.ycbcr_to_rgb(np.array(img_pil)))
            img = np.array(img_pil)
        else:
            final_img = canvas.astype(np.uint8)

        
        final_img_pil = Image.fromarray(final_img)
        if downsample_rate > 1:
            final_img_pil = cls.upsample_image(final_img_pil, original_size, downsample_alg=downsample_alg)
            img = np.array(img_pil)
            final_img = np.array(final_img_pil)

        total_bits = len(bitstream)

        losses = []
        for channel_idx in range(3):
            full_diff = img[:, :, channel_idx].astype(np.float32) - final_img[:, :, channel_idx].astype(np.float32)
            mse = int(np.mean(full_diff ** 2, dtype=np.float32))
            losses.append(mse)

        orig_size = img_pil.size[0]*img_pil.size[1]*3*8
        bit_stats = f"\n========================\nBITSTREAM STATS:\n"
        bit_stats += f"Header: {header_bits} bits\n"
        bit_stats += f"Strokes: {total_bits - header_bits} bits\n"
        bit_stats += f"Total: {total_bits/8/1024:.2f} KB from {orig_size/1024/8:.2f} KB original ({(total_bits/orig_size)*100:.2f}% size, {orig_size/total_bits:.2f}x compression)\n"

        # BINARY SAVE
        if save_filename:
            try:
                with open(save_filename, "wb") as f:
                    b_data, pad_len = cls._bits_to_bytes(bitstream)
                    f.write(bytes([pad_len])) 
                    f.write(b_data)
                bit_stats += f"Binary bitstream saved to {save_filename}\n"
            except Exception as e:
                bit_stats += f"Error saving bitstream: {e}\n"
        else:
            bit_stats += f"Bitstream not saved (save_filename=None).\n"
        
        bit_stats += f"========================\n"

        res = [final_img_pil, bit_stats, bitstream, losses]

        return (*res,)
    
    @classmethod
    def compress_from_filename(cls, filename, **kwargs):
        """Compresses an image from a given filename using the compress method. """
        try:
            img_pil = Image.open(filename)
        except Exception as e:
            return None, f"Error loading image: {e}"
        
        return cls.compress(img_pil, **kwargs)

    @classmethod
    def decompress(cls, bitstream, verbose=False, use_numba=True):
        """ Decompresses a PBC bitstream back into an image. """
        # Downsample flag
        downsample_flag = bitstream[0]
        read_i = 1
        if downsample_flag == '1':
            original_w = cls.decode_int(bitstream[read_i:read_i+16], signed=False)
            read_i += 16
            original_h = cls.decode_int(bitstream[read_i:read_i+16], signed=False)
            read_i += 16
            if verbose:
                print(f"Downsampled image detected. Original dimensions: {original_h}x{original_w}")
        else:
            original_w, original_h = -1, -1
            if verbose:
                print("No downsampling detected.")

        # YCbCr flag
        ycbcr_flag = bitstream[read_i]
        read_i += 1
        if verbose:
            print(f"Color space bit: {ycbcr_flag} ({'YCbCr' if ycbcr_flag == '1' else 'RGB'})")
        # Image dimensions
        h = cls.decode_int(bitstream[read_i:read_i+16], signed=False)
        read_i += 16
        w = cls.decode_int(bitstream[read_i:read_i+16], signed=False)
        read_i += 16
        if verbose:
            print(f"Image dimensions: {h}x{w}")
        # Stroke count
        stroke_count = cls.decode_int(bitstream[read_i:read_i+20], signed=False)
        read_i += 20
        if verbose:
            print(f"Stroke count: {stroke_count}")
        # Start color
        start_r = cls.decode_int(bitstream[read_i:read_i+8], signed=False)
        read_i += 8
        start_g = cls.decode_int(bitstream[read_i:read_i+8], signed=False)
        read_i += 8
        start_b = cls.decode_int(bitstream[read_i:read_i+8], signed=False)
        read_i += 8
        start_color = (start_r, start_g, start_b)
        if verbose:
            if ycbcr_flag == '1':
                print(f"Start color: Y={start_r}, Cb={start_g}, Cr={start_b}")
            else:
                print(f"Start color: R={start_r}, G={start_g}, B={start_b}")
        # Initialize canvas
        canvas = np.full((h, w, 3), start_color, dtype=np.int16)
        # Downsample initialize flag
        downsample_initialize_flag = bitstream[read_i]
        read_i += 1
        if verbose:
            print(f"Downsample initialize flag: {downsample_initialize_flag} ({'Enabled' if downsample_initialize_flag == '1' else 'Disabled'})")
        if downsample_initialize_flag == "1":
            n_h = cls.decode_int(bitstream[read_i:read_i+10], signed=False)
            read_i += 10
            n_w = cls.decode_int(bitstream[read_i:read_i+10], signed=False)
            read_i += 10
            if verbose:
                print(f"Downsample initialize dimensions: {n_h}x{n_w} | Rate: {h/n_h:.2f}")
            canvas_data_bits = bitstream[read_i:read_i+(n_h * n_w * 3 * 8)]
            if verbose:
                print(f"len(canvas_data_bits): {len(canvas_data_bits)}")
            canvas = cls.bitstream_to_array(canvas_data_bits, n_h, n_w, channels=3)
            canvas = np.array(Image.fromarray(canvas).resize((w, h), Image.LANCZOS), dtype=np.int16)
            if ycbcr_flag == '1':
                canvas = cls.rgb_to_ycbcr(canvas).astype(np.int16)
            read_i += (n_h * n_w * 3 * 8)

        # Size start and end
        size_start = cls.decode_int(bitstream[read_i:read_i+16], signed=False)
        read_i += 16
        size_end = cls.decode_int(bitstream[read_i:read_i+16], signed=False)
        read_i += 16
        if verbose:
            print(f"Size start: {size_start}, Size end: {size_end}")
            print(f"Equivalent to input fractions: start={((size_start - 2)/(min(h, w) - 2)):.4f}, end={((size_end - 2)/(min(h, w) - 2)):.4f}")
        # Decay parameters
        # Cutoff
        cutoff_flag = bitstream[read_i]
        read_i += 1
        if verbose:
            print(f"Cutoff flag: {cutoff_flag} ({'Auto' if cutoff_flag == '1' else 'Custom'})")
        if cutoff_flag == '0':
            cutoff_bits = bitstream[read_i:read_i+20]
            cutoff = cls.decode_float(cutoff_bits, decimals=4)
        else:
            cutoff = -1.0
        cutoff = round(cutoff, 4)
        read_i += 20
        # Softness
        softness_flag = bitstream[read_i]
        read_i += 1
        if verbose:
            print(f"Softness flag: {softness_flag} ({'Auto' if softness_flag == '1' else 'Custom'})")
        if softness_flag == '0':
            softness_bits = bitstream[read_i:read_i+20]
            softness = cls.decode_float(softness_bits, decimals=4)
        else:
            softness = -1.0
        softness = round(softness, 4)
        read_i += 20
        # Progress
        progress_flag = bitstream[read_i]
        read_i += 1
        if verbose:
            print(f"Progress flag: {progress_flag} ({'Auto' if progress_flag == '1' else 'Custom'})")
        if progress_flag == '0':
            progress_bits = bitstream[read_i:read_i+20]
            progress = cls.decode_float(progress_bits, decimals=4)
        else:
            progress = -1.0
        progress = round(progress, 4)
        read_i += 20
        # Get sizes from decay curve
        sizes, _ = cls.get_decay_curve(stroke_count, size_start, size_end, cutoff, softness, progress)
        if verbose:
            print(f"Decay parameters: cutoff={cutoff:.4f}, softness={softness:.4f}, progress={progress:.4f}")
        # Mult list length
        len_multlist = cls.decode_int(bitstream[read_i:read_i+9], signed=False)
        read_i += 9
        mult_list = []
        # Mult list
        for _ in range(len_multlist):
            mult_bits = bitstream[read_i:read_i+9]
            mult = cls.decode_int(mult_bits, signed=True)
            mult_list.append(mult)
            read_i += 9
        mult_arr = np.array(mult_list)
        multlist_len = len(mult_arr)
        multlist_bitcount = int(np.ceil(np.log2(multlist_len)))
        if multlist_len != len_multlist:
            print(f"ERROR: Mult list length mismatch during decoding. Expected {len_multlist}, got {multlist_len}.")
            return -1
        if verbose:
            print(f"Multiplier list: {mult_list}")
            print(f"Multiplier list length: {multlist_len} | Bitcount per multiplier index: {multlist_bitcount}")
        # Quadrant warmup time
        quadrant_warmup_time = cls.decode_int(bitstream[read_i:read_i+20], signed=False)
        read_i += 20
        # Strokes per quadrant
        strokes_per_quadrant = cls.decode_int(bitstream[read_i:read_i+20], signed=False)
        read_i += 20
        # Quadrant max bits
        quadrant_max_bits = cls.decode_int(bitstream[read_i:read_i+8], signed=False)
        read_i += 8
        # Quadrant padding
        quadrant_padding = cls.decode_int(bitstream[read_i:read_i+8], signed=False)
        read_i += 8
        if verbose:
            print(f"Quadrant Parameters: warmup_time={quadrant_warmup_time}, strokes_per_quadrant={strokes_per_quadrant}, max_bits={quadrant_max_bits}, padding={quadrant_padding}")
        # Channel cycle flag
        channel_cycle_flag = bitstream[read_i]
        strokes_per_channel_cycle = None
        channel_cycle_warmup_time = None
        read_i += 1
        if verbose:
            print(f"Channel cycle flag: {channel_cycle_flag} ({'Enabled' if channel_cycle_flag == '1' else 'Disabled'})")
        if channel_cycle_flag == '1':
            # Strokes per channel cycle
            strokes_per_channel_cycle = cls.decode_int(bitstream[read_i:read_i+20], signed=False)
            read_i += 20
            # Channel cycle warmup time
            channel_cycle_warmup_time = cls.decode_int(bitstream[read_i:read_i+20], signed=False)
            read_i += 20
            if verbose:
                print(f"Channel Cycle Parameters: strokes_per_channel_cycle={strokes_per_channel_cycle}, warmup_time={channel_cycle_warmup_time}")
        # HEADER READ COMPLETE, NOW PREPARING MAIN LOOP VARIABLES

        quadrant_switch_counters = [quadrant_warmup_time//3, quadrant_warmup_time//3, quadrant_warmup_time//3]
        quadrant_inputs = [None, None, None]
        channel_selector = [0, 1, 2]
        channel_cycle_timer = channel_cycle_warmup_time if channel_cycle_warmup_time is not None else 0

        rng_state = None
        if use_numba:
            rng_state = np.uint64(2003)

        if verbose:
            print("\n======================================================\nBeginning main decompression loop...")
        for i in range(stroke_count):
            channel_idx = channel_selector[i % 3]
            canvas_layer = canvas[:, :, channel_idx]
            size = sizes[i]

            if quadrant_switch_counters[channel_idx] <= 0:
                quadrant_bitcount = cls.get_quadrant_bitcount(h, w, size, quadrant_max_bits)
                quadrant_input_bits = bitstream[read_i:read_i+quadrant_bitcount]
                quadrant_inputs[channel_idx] = quadrant_input_bits
                read_i += quadrant_bitcount
                quadrant_switch_counters[channel_idx] = strokes_per_quadrant

            if channel_cycle_flag == '1':
                if channel_cycle_timer <= 0:
                    new_channel_selector = []
                    for _ in range(3):
                        ch_bits = bitstream[read_i:read_i+2]
                        ch = cls.decode_int(ch_bits, signed=False)
                        new_channel_selector.append(ch)
                        read_i += 2
                    channel_selector = new_channel_selector
                    channel_cycle_timer = strokes_per_channel_cycle

            if use_numba:
                # 1. PRE-CALCULATE BOUNDARIES IN PYTHON
                # This avoids passing Strings/None to Numba
                if quadrant_inputs[channel_idx] is None:
                    r_s, c_s = 0, 0
                    r_e, c_e = h - size, w - size
                else:
                    q_int = cls.decode_int(quadrant_inputs[channel_idx], signed=False)
                    q_bitcount = len(quadrant_inputs[channel_idx])
                    r_s, c_s, r_e, c_e = process_quadrant_int(h, w, size, q_int, q_bitcount, quadrant_padding)

                rng_state, row, col = get_stroke_coords_rolling(rng_state, r_s, c_s, r_e, c_e)
                rng_state = np.uint64(rng_state)
            else:
                row, col = cls.get_stroke_params(i, h, w, size, quadrant_input=quadrant_inputs[channel_idx], quadrant_padding=quadrant_padding)


            if use_numba:
                stroke_indices = []
                for _ in range(4):
                    mult_idx_bits = bitstream[read_i:read_i+multlist_bitcount]
                    mult_idx = cls.decode_int(mult_idx_bits, signed=False)
                    stroke_indices.append(mult_idx)
                    read_i += multlist_bitcount 
                canvas_layer = stroke_numba_decompress(canvas_layer, h, w, row, col, size, mult_arr, stroke_indices)
            else:
                half = size // 2
                slices = [
                    (slice(row, row+half), slice(col, col+half)),
                    (slice(row, row+half), slice(col+half, col+size)),
                    (slice(row+half, row+size), slice(col, col+half)),
                    (slice(row+half, row+size), slice(col+half, col+size))
                ]
                
                for sl in slices:
                    if sl[0].start >= h or sl[1].start >= w:
                        continue
                    if sl[0].start >= sl[0].stop or sl[1].start >= sl[1].stop:
                        continue
                    
                    mult_idx_bits = bitstream[read_i:read_i+multlist_bitcount]
                    mult_idx = cls.decode_int(mult_idx_bits, signed=False)
                    read_i += multlist_bitcount
                    
                    best_mult = mult_arr[mult_idx]

                    canvas_layer[sl] = np.clip(canvas_layer[sl] + best_mult, 0, 255)

            quadrant_switch_counters[channel_idx] -= 1
            if channel_cycle_flag == '1':
                channel_cycle_timer -= 1

        if downsample_flag == "1":
            if verbose:
                print(f"Upsampling final image to original dimensions: {original_h}x{original_w}")
            canvas_pil = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
            canvas_pil = cls.upsample_image(canvas_pil, (original_w, original_h), downsample_alg=Image.BICUBIC)
            canvas = np.array(canvas_pil, dtype=np.int16)


        canvas = np.clip(canvas, 0, 255)
        if ycbcr_flag == '1':
            final_img = cls.ycbcr_to_rgb(canvas)
            #final_img = canvas.astype(np.uint8)
        else:
            final_img = canvas.astype(np.uint8)
        return Image.fromarray(final_img)

    @classmethod
    def decompress_from_filename(cls, filename, verbose=False, use_numba=True):
        """ Decompresses a PBC binary file back into an image. """
        try:
            with open(filename, "rb") as f:
                pad_len = int.from_bytes(f.read(1), byteorder='big')
                byte_data = f.read()
            bitstream = cls._bytes_to_bits(byte_data, pad_len)
        except Exception as e:
            print(f"Error reading bitstream from {filename}: {e}")
            return None
        return cls.decompress(bitstream, verbose=verbose, use_numba=use_numba)

    @classmethod
    def compress_stream(cls, img_pil, stroke_count=-1, size_range=(-1, -1), mult_list=[-10, 0, 5, 20], start_mode="Average",
                    start_custom=(128, 128, 128), decay_params={'cutoff': -1, 'softness': -1, 'progress': -1},
                    strokes_per_quadrant=100, quadrant_warmup_time=-1, quadrant_max_bits=8, quadrant_padding=4, quadrant_selection_criteria="Sum",
                    channel_cycle="Smart", strokes_per_channel_cycle=100, channel_cycle_warmup_time=0.9, cycle_selection_criteria="Min",
                    color_space="RGB", downsample_rate=-1, display_autos=False,
                    save_filename=-1, use_numba=True, stream_interval=100, downsample_initialize=True, downsample_initialize_rate=16, downsample_alg=Image.BICUBIC):
        """Generator version of compress that yields intermediate bitstreams."""

        if img_pil is None:
            return None, "Please upload an image first."
        
        if color_space == "YCbCr":
            img_pil = Image.fromarray(np.array(img_pil.convert("YCbCr")))
        
        if save_filename == -1:
            save_filename = f"compressed.pbc"

        bitstream = ""
        original_size = img_pil.size
        ori_w, ori_h = original_size

        if downsample_rate == -1:
            if min(original_size) < 600:
                downsample_rate = 1
            else:
                downsample_rate = min(original_size) / 500
                if display_autos:
                    print(f'Auto-calculated downsample_rate: {downsample_rate}')

        if downsample_rate > 1:
            img_pil_downsampled = cls.downsample_image(img_pil, downsample_rate, downsample_alg=downsample_alg)
            img = np.array(img_pil_downsampled, dtype=np.int16)
            # (HEADER BITS) downsample flag (1=downsampled)
            bitstream += "1"
            # original width and height bits (16 bits each)
            bitstream += cls.encode_int(ori_w, signed=False, bitcount=16)
            bitstream += cls.encode_int(ori_h, signed=False, bitcount=16)
        else:
            img = np.array(img_pil, dtype=np.int16)
            # (HEADER BITS) downsample flag (0=not downsampled)
            bitstream += "0"

        if color_space == "YCbCr":
            # (HEADER BITS) color space bit YCbCr=1
            bitstream += "1"
        else:
            # (HEADER BITS) color space bit RGB=0
            bitstream += "0"

        h, w = img.shape[:2]

        if stroke_count == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 20000
            b = 0.0015
            c = 3200
            stroke_count = int(a + b * ((max(original_size) + c) ** 2))
            if display_autos:
                print(f'Auto-calculated stroke_count: {stroke_count} (Based on {original_size[0]}x{original_size[1]})')

        # (HEADER BITS) image size bits (max image height and width allowed is 65535 (16 bits each), should be enough for any reasonable use case)
        bitstream += cls.encode_int(h, signed=False, bitcount=16) # unsigned, obviously
        bitstream += cls.encode_int(w, signed=False, bitcount=16) # unsigned, obviously
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        # (HEADER BITS) stroke count bits (stroke counts above 1M probably unfeasible, 20 bits should be enough)
        bitstream += cls.encode_int(stroke_count, signed=False, bitcount=20) # unsigned, obviously
        
        if start_mode == "Black": start_color = (0, 0, 0) if color_space == "RGB" else (0, 128, 128)
        elif start_mode == "White": start_color = (255, 255, 255) if color_space == "RGB" else (255, 128, 128)
        elif start_mode == "Custom": start_color = start_custom
        elif start_mode == "Average" or start_mode == "Mean": start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
        elif start_mode == "Median": start_color = (int(np.median(r)), int(np.median(g)), int(np.median(b)))
        elif start_mode == "True Median": start_color = np.median(img.reshape(-1, 3), axis=0).astype(int)
        elif start_mode == "Random": start_color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
        else:
            print(f'Warning: Unknown start_mode "{start_mode}", defaulting to "Average".')
            start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))

        if display_autos:
            print(f'Start color selected: R={start_color[0]}, G={start_color[1]}, B={start_color[2]}')

        # (HEADER BITS) start color bits (0-255 for each channel, 8 bits each)
        bitstream += cls.encode_int(start_color[0], signed=False, bitcount=8) # R
        bitstream += cls.encode_int(start_color[1], signed=False, bitcount=8) # G
        bitstream += cls.encode_int(start_color[2], signed=False, bitcount=8) # B


        if downsample_initialize:
            if downsample_initialize_rate < 32:
                if stroke_count > 20000:
                    if decay_params["cutoff"] == -1:
                        decay_params["cutoff"] = 0.3
                    if size_range == (-1, -1):
                        size_range = (0.05, 0.01)
                    if quadrant_warmup_time == -1:
                        quadrant_warmup_time = 0.1
                else:
                    if decay_params["cutoff"] == -1:
                        decay_params["cutoff"] = 0.7
                    if size_range == (-1, -1):
                        size_range = (0.1, 0.03)
                    if quadrant_warmup_time == -1:
                        quadrant_warmup_time = 0.7

        if downsample_initialize:
            bitstream += "1" # downsample initialize flag bit
            n_h, n_w = int(h/downsample_initialize_rate), int(w/downsample_initialize_rate)
            bitstream += cls.encode_int(n_h, signed=False, bitcount=10) # downsampled height bits
            bitstream += cls.encode_int(n_w, signed=False, bitcount=10) # downsampled width bits
            if downsample_rate > 1:
                canvas = np.array(img_pil_downsampled.resize((n_w, n_h), downsample_alg), dtype=np.uint8)
            else:
                canvas = np.array(img_pil.resize((n_w, n_h), downsample_alg), dtype=np.uint8)
            #bitstream += "1" * (n_h * n_w * 3 * 8)
            if color_space == "YCbCr":
                canvas = cls.ycbcr_to_rgb(canvas)
            bitstream += cls.array_to_bitstream(canvas)
            canvas = np.array(Image.fromarray(canvas).resize((w, h), downsample_alg), dtype=np.int16)
            if color_space == "YCbCr":
                canvas = cls.rgb_to_ycbcr(canvas).astype(np.int16)
            if display_autos:
                print(f'Downsample initialize test enabled with rate {downsample_initialize_rate}, canvas initialized from downsampled image.')
        else:
            canvas = np.full((h, w, 3), start_color, dtype=np.int16)
            bitstream += "0" # downsample initialize flag bit

        size_start = size_range[0]
        size_end = size_range[1]
        if size_start == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.3
            b = 1.00095
            c = 7000
            size_start = a + (1/b)**(stroke_count + c)
            if display_autos:
                print(f"Auto-calculated size start: {size_start:.3f}")
        if size_end == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.01
            b = 1.00015
            c = 10200
            size_end = a + (1/b)**(stroke_count + c)
            if display_autos:
                print(f"Auto-calculated size end: {size_end:.3f}")

        size_start = int(size_start * (min(h, w) - 2)) + 2
        size_end = int(size_end * (min(h, w) - 2)) + 2

        if display_autos:
            print(f"Calculated size range: {size_start} to {size_end} pixels.")

        # maximum and minimum sizes are the same bitcount as the height and width, since size cannot be larger than image dimensions
        # (HEADER BITS) size start and end bits
        bitstream += cls.encode_int(size_start, signed=False, bitcount=16)
        bitstream += cls.encode_int(size_end, signed=False, bitcount=16)

        sizes, decay_params_encoding = cls.get_decay_curve(stroke_count, size_start, size_end,
                                    decay_params['cutoff'], decay_params['softness'], decay_params['progress'], display_autos=display_autos)
        
        # (HEADER BITS) decay parameters bits
        bitstream += decay_params_encoding

        encoded_strokes = []
        mult_arr = np.array(mult_list)
        multlist_len = len(mult_arr)
        multlist_bitcount = int(np.ceil(np.log2(multlist_len)))
        # (HEADER BITS) multlist length bits
        # since we need to know how many multipliers there are to decode them later, there's no ending delimiter
        bitstream += cls.encode_int(multlist_len, signed=False, bitcount=9) # max amount of multipliers possible: (-255, 255) so 512 multipliers, so 9 bits for the number
        # (HEADER BITS) multlist bits
        for mult in mult_arr:
            bitstream += cls.encode_int(mult, signed=True, bitcount=9) # signed, since multipliers can be negative, 9 bits cover the -255 to 255 range

        if quadrant_warmup_time == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.75
            b = 0.000014
            c = -1000
            d = 550000
            quadrant_warmup_time = a - (b*(stroke_count + c) ** 2) / d
            quadrant_warmup_time = max(0.0, min(quadrant_warmup_time, 1.0))
            if display_autos:
                print(f'Auto-calculated quadrant_warmup_time: {quadrant_warmup_time:.4f}')


        quadrant_warmup_time = int(quadrant_warmup_time * stroke_count)
        if display_autos:
            print(f'Calculated quadrant_warmup_time: {quadrant_warmup_time} strokes.')

        # (HEADER BITS) quadrant warmup time bits
        bitstream += cls.encode_int(quadrant_warmup_time, signed=False, bitcount=20)
        # (HEADER BITS) strokes per quadrant bits
        bitstream += cls.encode_int(strokes_per_quadrant, signed=False, bitcount=20)
        # (HEADER BITS) quadrant max bits bits
        bitstream += cls.encode_int(quadrant_max_bits, signed=False, bitcount=8) # max 255 bits should be way more than enough, default is 8
        # (HEADER BITS) quadrant padding bits
        bitstream += cls.encode_int(quadrant_padding, signed=False, bitcount=8) # max 255 pixel padding should be way more than enough, default is 4

        quadrant_switch_counters = [quadrant_warmup_time//3, quadrant_warmup_time//3, quadrant_warmup_time//3]
        quadrant_inputs = [None, None, None]

        if not channel_cycle:
            channel_cycle = None
        channel_selector = [0, 1, 2] # R=0, G=1, B=2
        channel_cycle_timer = int(channel_cycle_warmup_time * stroke_count)
        # by default its 0 1 2 to cycle through 3 channels
        # but it can be modified dynamically to prioritize certain channels with more error
        # for example [0, 0, 1] would be that it does R R G R R G ...

        cls.plot_curve_plt(size_start, size_end, stroke_count, decay_params['softness'], decay_params['progress'], decay_params['cutoff'], q_warmup=quadrant_warmup_time, cycle_warmup=channel_cycle_timer)

        # (HEADER BITS) channel cycle bool bit
        bitstream += "1" if channel_cycle else "0"
        if channel_cycle:
            # (HEADER BITS) strokes per channel cycle bits
            bitstream += cls.encode_int(strokes_per_channel_cycle, signed=False, bitcount=20)
            # (HEADER BITS) channel cycle warmup time bits
            bitstream += cls.encode_int(int(channel_cycle_warmup_time * stroke_count), signed=False, bitcount=20)

        header_bits = len(bitstream)
        # print(f"Header bits: {header_bits} bits")
        # (MAIN LOOP)

        rng_state = None
        if use_numba:
            # Initialize Numba RNG state
            rng_state = np.uint64(2003)

        stream_timer = 0

        for i in range(stroke_count):
            channel_idx = channel_selector[i % 3]
            target_layer = img[:, :, channel_idx]
            canvas_layer = canvas[:, :, channel_idx]
            size = sizes[i]

            if quadrant_switch_counters[channel_idx] <= 0:
                error_layer = np.abs(target_layer - canvas_layer)
                quadrant_bitcount = cls.get_quadrant_bitcount(h, w, size, quadrant_max_bits)
                quadrant_inputs[channel_idx] = cls.select_quadrant(error_layer, quadrant_bitcount, selection_criteria=quadrant_selection_criteria)
                # (MAIN LOOP BITS) quadrant bits
                bitstream += quadrant_inputs[channel_idx]
                quadrant_switch_counters[channel_idx] = strokes_per_quadrant

            if channel_cycle:
                if channel_cycle_timer <= 0:
                    # Re-evaluate channel priorities based on current error
                    full_error_layer = np.abs(img - canvas)
                    channel_selector = cls.channel_cycle_strategy(full_error_layer, strategy=channel_cycle, selection_criteria=cycle_selection_criteria)
                    # (MAIN LOOP BITS) channel cycle bits
                    for ch in channel_selector:
                        bitstream += cls.encode_int(ch, signed=False, bitcount=2) # 2 bits to represent 0-2
                    channel_cycle_timer = strokes_per_channel_cycle

            # RETURNS ROW, COL
            if use_numba:
                # 1. PRE-CALCULATE BOUNDARIES IN PYTHON
                # This avoids passing Strings/None to Numba
                if quadrant_inputs[channel_idx] is None:
                    r_s, c_s = 0, 0
                    r_e, c_e = h - size, w - size
                else:
                    q_int = cls.decode_int(quadrant_inputs[channel_idx], signed=False)
                    q_bitcount = len(quadrant_inputs[channel_idx])
                    r_s, c_s, r_e, c_e = process_quadrant_int(h, w, size, q_int, q_bitcount, quadrant_padding)
                rng_state, row, col = get_stroke_coords_rolling(rng_state, r_s, c_s, r_e, c_e)
                rng_state = np.uint64(rng_state)
            else:
                row, col = cls.get_stroke_params(i, h, w, size, quadrant_input=quadrant_inputs[channel_idx], quadrant_padding=quadrant_padding)

            if use_numba:
                stroke_indices, canvas_layer = stroke_numba(target_layer, canvas_layer, h, w, row, col, size, mult_arr)
            else:
                half = size // 2
                # Slices are [Row, Col]
                slices = [
                    (slice(row, row+half), slice(col, col+half)),
                    (slice(row, row+half), slice(col+half, col+size)),
                    (slice(row+half, row+size), slice(col, col+half)),
                    (slice(row+half, row+size), slice(col+half, col+size))
                ]
                
                stroke_indices = []
                for sl in slices:
                    # Basic boundary check (Never supposed to happen, but just in case)
                    if sl[0].start >= h or sl[1].start >= w:
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nSlice start out of bounds: {sl}, image size: ({h}, {w})\n----------\n\n\n")
                        continue

                    if sl[0].start >= sl[0].stop or sl[1].start >= sl[1].stop:
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nInvalid slice with start >= stop: {sl}\n----------\n\n\n")
                        continue
                    
                    # Extract region
                    target_slice = target_layer[sl]
                    canvas_slice = canvas_layer[sl]

                    # Another safety check for empty slices (which should not happen)
                    if target_slice.size == 0: 
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nEmpty slice encountered: {sl}\n----------\n\n\n")
                        continue

                    diff = target_slice - canvas_slice
                    mean_diff = np.mean(diff)
                    
                    best_idx = np.argmin(np.abs(mult_arr - mean_diff))
                    best_mult = mult_arr[best_idx]
                    
                    canvas_layer[sl] = np.clip(canvas_layer[sl] + best_mult, 0, 255)
                    stroke_indices.append(best_idx)

            # (MAIN LOOP BITS) stroke indices bits
            for idx in stroke_indices:
                bitstream += cls.encode_int(idx, signed=False, bitcount=multlist_bitcount)

            encoded_strokes.append(stroke_indices)
            quadrant_switch_counters[channel_idx] -= 1
            if channel_cycle:
                channel_cycle_timer -= 1

            if stream_timer <= 0:
                stream_timer = stream_interval - 1
                interim_img = np.clip(canvas, 0, 255).astype(np.uint8)
                if color_space == "YCbCr":
                    interim_img = cls.ycbcr_to_rgb(interim_img)
                interim_losses = [int(np.mean((interim_img[:, :, channel_idx].astype(np.float32) - img[:, :, channel_idx].astype(np.float32)) ** 2)) for channel_idx in range(3)]
                yield (Image.fromarray(interim_img), f"Processed {i}/{stroke_count} strokes. {((i)/stroke_count)*100:.2f}%", len(bitstream), interim_losses)
            else:
                stream_timer -= 1

        canvas = np.clip(canvas, 0, 255)

        if color_space == "YCbCr":
            final_img = cls.ycbcr_to_rgb(canvas).astype(np.uint8)
            img = np.array(img_pil)
        else:
            final_img = canvas.astype(np.uint8)

        
        final_img_pil = Image.fromarray(final_img)
        if downsample_rate > 1:
            final_img_pil = cls.upsample_image(final_img_pil, original_size, downsample_alg=downsample_alg)
            img = np.array(img_pil)
            final_img = np.array(final_img_pil)

        total_bits = len(bitstream)

        losses = []
        for channel_idx in range(3):
            full_diff = img[:, :, channel_idx].astype(np.float32) - final_img[:, :, channel_idx].astype(np.float32)
            mse = int(np.mean(full_diff ** 2, dtype=np.float32))
            losses.append(mse)

        orig_size = img_pil.size[0]*img_pil.size[1]*3*8
        bit_stats = f"\n========================\nBITSTREAM STATS:\n"
        bit_stats += f"Header: {header_bits} bits\n"
        bit_stats += f"Strokes: {total_bits - header_bits} bits\n"
        bit_stats += f"Total: {total_bits/8/1024:.2f} KB from {orig_size/1024/8:.2f} KB original ({(total_bits/orig_size)*100:.2f}% size, {orig_size/total_bits:.2f}x compression)\n"

        # BINARY SAVE
        if save_filename:
            try:
                with open(save_filename, "wb") as f:
                    b_data, pad_len = cls._bits_to_bytes(bitstream)
                    f.write(bytes([pad_len])) 
                    f.write(b_data)
                bit_stats += f"Binary bitstream saved to {save_filename}\n"
            except Exception as e:
                bit_stats += f"Error saving bitstream: {e}\n"
        else:
            bit_stats += f"Bitstream not saved (save_filename=None).\n"
        
        bit_stats += f"========================\n"

        res = [final_img_pil, bit_stats, total_bits, losses]
        yield (*res,)
        return (*res,)

    @classmethod
    def compress_and_generate_video(cls, img, fps=60, stream_interval=-1, 
         save_filename="compressed_output.pbc", video_filename="compression_evolution.mp4", **compress_kwargs):
        """
        Compress an image using PBC and generate a video showing the compression evolution.
        """
        # Collect all interim compressed images
        frames = []
        loss_history = [[] for _ in range(4)]  # R, G, B, Average

        stroke_count = compress_kwargs.get('stroke_count', -1)
        if stroke_count == -1:
            # Auto-calculate stroke_count if default
            original_size = img.size
            a = 20000
            b = 0.0015
            c = 3200
            stroke_count = int(a + b * ((max(original_size) + c) ** 2))

        if stream_interval == -1:
            # adjust stream_interval to fit 5 seconds with given fps
            stream_interval = stroke_count // (fps * 4)

        # Use a for loop to iterate over the generator
        for interim_result in cls.compress_stream(
            img,
            save_filename=save_filename,
            stream_interval=stream_interval,
            **compress_kwargs
        ):
            compressed_img, bit_info, total_bits, losses = interim_result
            avg_loss = (losses[0] + losses[1] + losses[2]) / 3

            # Print progress
            interim_print = (
                f"Progress update: {bit_info} | Size: {total_bits / 8 / 1024:.2f} KB\n"
                f"Losses: {losses}\n"
                f"Average MSE: {avg_loss:.2f}\n"
            )
            print(interim_print, end="\r", flush=True)

            # Append losses
            loss_history[0].append(losses[0])
            loss_history[1].append(losses[1])
            loss_history[2].append(losses[2])
            loss_history[3].append(avg_loss)

            # Convert PIL image to OpenCV format and add to frames
            frame = cv2.cvtColor(np.array(compressed_img), cv2.COLOR_RGB2BGR)
            frames.append(frame)

        # If you want the final result, you can collect it after the loop
        compressed_img, bit_info, total_bits, losses = interim_result
        print("\nFinal Results:")
        print(bit_info)
        print(losses)
        final_mse = int((losses[0] + losses[1] + losses[2]) / 3)
        print(f"Final MSE: {final_mse}")
        compression_rate = img.width * img.height * 3 * 8 / (total_bits)
        print(f"Final Size: {total_bits / 8 / 1024:.2f} KB | Compression Rate: {compression_rate:.2f}x")

        # Plot compression loss over time
        try:
            xticks = np.arange(0, stroke_count + 1, step=stream_interval)
            plt.plot(xticks, loss_history[0], label="Red Channel", color="red")
            plt.plot(xticks, loss_history[1], label="Green Channel", color="green")
            plt.plot(xticks, loss_history[2], label="Blue Channel", color="blue")
            plt.plot(xticks, loss_history[3], label="Average MSE", color="black", linestyle="--", linewidth=2)
            plt.xlabel("Stroke Count")
            plt.ylabel("Average MSE")
            plt.legend()
            plt.title(f"Compression Loss Over Time")
            plt.show()
        except ValueError:
            xticks = np.arange(0, stroke_count + 1, step=stream_interval)
            xticks = np.append(xticks, stroke_count)
            plt.plot(xticks, loss_history[0], label="Red Channel", color="red")
            plt.plot(xticks, loss_history[1], label="Green Channel", color="green")
            plt.plot(xticks, loss_history[2], label="Blue Channel", color="blue")
            plt.plot(xticks, loss_history[3], label="Average MSE", color="black", linestyle="--", linewidth=2)
            plt.xlabel("Stroke Count")
            plt.ylabel("Average MSE")
            plt.legend()
            plt.title(f"Compression Loss Over Time")
            plt.show()


        # Create a video from the collected frames
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for .mp4
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        for frame in frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved as {video_filename}")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(compressed_img)
        plt.axis('off')
        plt.title('Final Compressed Image')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original Image')
        plt.suptitle(f'Final MSE: {final_mse}\nCompression rate: {compression_rate:.2f}x', fontsize=16)
        plt.show()

    @classmethod
    def just_compress(cls, img_pil, stroke_count=-1, size_range=(-1, -1), mult_list=[-10, 0, 5, 20], start_mode="Average",
                 start_custom=(128, 128, 128), decay_params={'cutoff': -1, 'softness': -1, 'progress': -1},
                 strokes_per_quadrant=100, quadrant_warmup_time=-1, quadrant_max_bits=8, quadrant_padding=4, quadrant_selection_criteria="Sum",
                 channel_cycle="Smart", strokes_per_channel_cycle=100, channel_cycle_warmup_time=0.9, cycle_selection_criteria="Min",
                 color_space="RGB", downsample_rate=-1, display_autos=False, use_numba=True, downsample_initialize=True, downsample_initialize_rate=16, downsample_alg=Image.BICUBIC):
        """ 
        Function to just take an image and return the compressed bitstream
        without collecting additional info or constructing the image.
        """
        if img_pil is None:
            return None, "Please upload an image first."
        bitstream = ""
        original_size = img_pil.size
        ori_w, ori_h = original_size

        if color_space == "YCbCr":
            img_pil = Image.fromarray(np.array(img_pil.convert("YCbCr")))

        if downsample_rate == -1:
            if min(original_size) < 600:
                downsample_rate = 1
            else:
                downsample_rate = min(original_size) / 500
                if display_autos:
                    print(f'Auto-calculated downsample_rate: {downsample_rate}')

        if downsample_rate > 1:
            img_pil_downsampled = cls.downsample_image(img_pil, downsample_rate, downsample_alg=downsample_alg)
            img = np.array(img_pil_downsampled, dtype=np.int16)
            # (HEADER BITS) downsample flag (1=downsampled)
            bitstream += "1"
            # original width and height bits (16 bits each)
            bitstream += cls.encode_int(ori_w, signed=False, bitcount=16)
            bitstream += cls.encode_int(ori_h, signed=False, bitcount=16)
        else:
            img = np.array(img_pil, dtype=np.int16)
            # (HEADER BITS) downsample flag (0=not downsampled)
            bitstream += "0"

        if color_space == "YCbCr":
            # (HEADER BITS) color space bit YCbCr=1
            bitstream += "1"
        else:
            # (HEADER BITS) color space bit RGB=0
            bitstream += "0"

        h, w = img.shape[:2]

        if stroke_count == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 20000
            b = 0.0015
            c = 3200
            stroke_count = int(a + b * ((max(original_size) + c) ** 2))
            if display_autos:
                print(f'Auto-calculated stroke_count: {stroke_count} (Based on {original_size[0]}x{original_size[1]})')

        # (HEADER BITS) image size bits (max image height and width allowed is 65535 (16 bits each), should be enough for any reasonable use case)
        bitstream += cls.encode_int(h, signed=False, bitcount=16) # unsigned, obviously
        bitstream += cls.encode_int(w, signed=False, bitcount=16) # unsigned, obviously
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        # (HEADER BITS) stroke count bits (stroke counts above 1M probably unfeasible, 20 bits should be enough)
        bitstream += cls.encode_int(stroke_count, signed=False, bitcount=20) # unsigned, obviously
        
        if start_mode == "Black": start_color = (0, 0, 0) if color_space == "RGB" else (0, 128, 128)
        elif start_mode == "White": start_color = (255, 255, 255) if color_space == "RGB" else (255, 128, 128)
        elif start_mode == "Custom": start_color = start_custom
        elif start_mode == "Average" or start_mode == "Mean": start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
        elif start_mode == "Median": start_color = (int(np.median(r)), int(np.median(g)), int(np.median(b)))
        elif start_mode == "True Median": start_color = np.median(img.reshape(-1, 3), axis=0).astype(int)
        elif start_mode == "Random": start_color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
        else:
            print(f'Warning: Unknown start_mode "{start_mode}", defaulting to "Average".')
            start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))

        if display_autos:
            print(f'Start color selected: R={start_color[0]}, G={start_color[1]}, B={start_color[2]}')

        # (HEADER BITS) start color bits (0-255 for each channel, 8 bits each)
        bitstream += cls.encode_int(start_color[0], signed=False, bitcount=8) # R
        bitstream += cls.encode_int(start_color[1], signed=False, bitcount=8) # G
        bitstream += cls.encode_int(start_color[2], signed=False, bitcount=8) # B


        if downsample_initialize:
            if downsample_initialize_rate < 32:
                if stroke_count > 20000:
                    if decay_params["cutoff"] == -1:
                        decay_params["cutoff"] = 0.3
                    if size_range == (-1, -1):
                        size_range = (0.05, 0.01)
                    if quadrant_warmup_time == -1:
                        quadrant_warmup_time = 0.1
                else:
                    if decay_params["cutoff"] == -1:
                        decay_params["cutoff"] = 0.7
                    if size_range == (-1, -1):
                        size_range = (0.1, 0.03)
                    if quadrant_warmup_time == -1:
                        quadrant_warmup_time = 0.7

        if downsample_initialize:
            bitstream += "1" # downsample initialize flag bit
            n_h, n_w = int(h/downsample_initialize_rate), int(w/downsample_initialize_rate)
            bitstream += cls.encode_int(n_h, signed=False, bitcount=10) # downsampled height bits
            bitstream += cls.encode_int(n_w, signed=False, bitcount=10) # downsampled width bits
            if downsample_rate > 1:
                canvas = np.array(img_pil_downsampled.resize((n_w, n_h), downsample_alg), dtype=np.uint8)
            else:
                canvas = np.array(img_pil.resize((n_w, n_h), downsample_alg), dtype=np.uint8)
            if color_space == "YCbCr":
                canvas = cls.ycbcr_to_rgb(canvas)
            #bitstream += "1" * (n_h * n_w * 3 * 8)
            bitstream += cls.array_to_bitstream(canvas)
            canvas = np.array(Image.fromarray(canvas).resize((w, h), downsample_alg), dtype=np.int16)
            if color_space == "YCbCr":
                canvas = cls.rgb_to_ycbcr(canvas).astype(np.int16)
            if display_autos:
                print(f'Downsample initialize test enabled with rate {downsample_initialize_rate}, canvas initialized from downsampled image.')
        else:
            canvas = np.full((h, w, 3), start_color, dtype=np.int16)
            bitstream += "0" # downsample initialize flag bit

        size_start = size_range[0]
        size_end = size_range[1]
        if size_start == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.3
            b = 1.00095
            c = 7000
            size_start = a + (1/b)**(stroke_count + c)
            if display_autos:
                print(f"Auto-calculated size start: {size_start:.3f}")
        if size_end == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.01
            b = 1.00015
            c = 10200
            size_end = a + (1/b)**(stroke_count + c)
            if display_autos:
                print(f"Auto-calculated size end: {size_end:.3f}")

        size_start = int(size_start * (min(h, w) - 2)) + 2
        size_end = int(size_end * (min(h, w) - 2)) + 2

        if display_autos:
            print(f"Calculated size range: {size_start} to {size_end} pixels.")

        # maximum and minimum sizes are the same bitcount as the height and width, since size cannot be larger than image dimensions
        # (HEADER BITS) size start and end bits
        bitstream += cls.encode_int(size_start, signed=False, bitcount=16)
        bitstream += cls.encode_int(size_end, signed=False, bitcount=16)

        sizes, decay_params_encoding = cls.get_decay_curve(stroke_count, size_start, size_end,
                                    decay_params['cutoff'], decay_params['softness'], decay_params['progress'], display_autos=display_autos)
        
        # (HEADER BITS) decay parameters bits
        bitstream += decay_params_encoding

        encoded_strokes = []
        mult_arr = np.array(mult_list)
        multlist_len = len(mult_arr)
        multlist_bitcount = int(np.ceil(np.log2(multlist_len)))
        # (HEADER BITS) multlist length bits
        # since we need to know how many multipliers there are to decode them later, there's no ending delimiter
        bitstream += cls.encode_int(multlist_len, signed=False, bitcount=9) # max amount of multipliers possible: (-255, 255) so 512 multipliers, so 9 bits for the number
        # (HEADER BITS) multlist bits
        for mult in mult_arr:
            bitstream += cls.encode_int(mult, signed=True, bitcount=9) # signed, since multipliers can be negative, 9 bits cover the -255 to 255 range

        if quadrant_warmup_time == -1:
            # VALUES FROM TESTING OUT A FORMULA ON DESMOS TO FIT EXPERIMENTAL DATA
            a = 0.75
            b = 0.000014
            c = -1000
            d = 550000
            quadrant_warmup_time = a - (b*(stroke_count + c) ** 2) / d
            quadrant_warmup_time = max(0.0, min(quadrant_warmup_time, 1.0))
            if display_autos:
                print(f'Auto-calculated quadrant_warmup_time: {quadrant_warmup_time:.4f}')


        quadrant_warmup_time = int(quadrant_warmup_time * stroke_count)
        if display_autos:
            print(f'Calculated quadrant_warmup_time: {quadrant_warmup_time} strokes.')

        # (HEADER BITS) quadrant warmup time bits
        bitstream += cls.encode_int(quadrant_warmup_time, signed=False, bitcount=20)
        # (HEADER BITS) strokes per quadrant bits
        bitstream += cls.encode_int(strokes_per_quadrant, signed=False, bitcount=20)
        # (HEADER BITS) quadrant max bits bits
        bitstream += cls.encode_int(quadrant_max_bits, signed=False, bitcount=8) # max 255 bits should be way more than enough, default is 8
        # (HEADER BITS) quadrant padding bits
        bitstream += cls.encode_int(quadrant_padding, signed=False, bitcount=8) # max 255 pixel padding should be way more than enough, default is 4

        quadrant_switch_counters = [quadrant_warmup_time//3, quadrant_warmup_time//3, quadrant_warmup_time//3]
        quadrant_inputs = [None, None, None]

        if not channel_cycle:
            channel_cycle = None
        channel_selector = [0, 1, 2] # R=0, G=1, B=2
        channel_cycle_timer = int(channel_cycle_warmup_time * stroke_count)
        # by default its 0 1 2 to cycle through 3 channels
        # but it can be modified dynamically to prioritize certain channels with more error
        # for example [0, 0, 1] would be that it does R R G R R G ...

        # (HEADER BITS) channel cycle bool bit
        bitstream += "1" if channel_cycle else "0"
        if channel_cycle:
            # (HEADER BITS) strokes per channel cycle bits
            bitstream += cls.encode_int(strokes_per_channel_cycle, signed=False, bitcount=20)
            # (HEADER BITS) channel cycle warmup time bits
            bitstream += cls.encode_int(int(channel_cycle_warmup_time * stroke_count), signed=False, bitcount=20)

        header_bits = len(bitstream)
        # print(f"Header bits: {header_bits} bits")
        # (MAIN LOOP)

        rng_state = None
        if use_numba:
            # Initialize Numba RNG state
            rng_state = np.uint64(2003)

        for i in range(stroke_count):
            channel_idx = channel_selector[i % 3]
            target_layer = img[:, :, channel_idx]
            canvas_layer = canvas[:, :, channel_idx]
            size = sizes[i]

            if quadrant_switch_counters[channel_idx] <= 0:
                error_layer = np.abs(target_layer - canvas_layer)
                quadrant_bitcount = cls.get_quadrant_bitcount(h, w, size, quadrant_max_bits)
                quadrant_inputs[channel_idx] = cls.select_quadrant(error_layer, quadrant_bitcount, selection_criteria=quadrant_selection_criteria)
                # (MAIN LOOP BITS) quadrant bits
                bitstream += quadrant_inputs[channel_idx]
                quadrant_switch_counters[channel_idx] = strokes_per_quadrant
            if channel_cycle:
                if channel_cycle_timer <= 0:
                    # Re-evaluate channel priorities based on current error
                    full_error_layer = np.abs(img - canvas)
                    channel_selector = cls.channel_cycle_strategy(full_error_layer, strategy=channel_cycle, selection_criteria=cycle_selection_criteria)
                    # (MAIN LOOP BITS) channel cycle bits
                    for ch in channel_selector:
                        bitstream += cls.encode_int(ch, signed=False, bitcount=2) # 2 bits to represent 0-2
                    channel_cycle_timer = strokes_per_channel_cycle

            # RETURNS ROW, COL
            if use_numba:
                # 1. PRE-CALCULATE BOUNDARIES IN PYTHON
                # This avoids passing Strings/None to Numba
                if quadrant_inputs[channel_idx] is None:
                    r_s, c_s = 0, 0
                    r_e, c_e = h - size, w - size
                else:
                    q_int = cls.decode_int(quadrant_inputs[channel_idx], signed=False)
                    q_bitcount = len(quadrant_inputs[channel_idx])
                    r_s, c_s, r_e, c_e = process_quadrant_int(h, w, size, q_int, q_bitcount, quadrant_padding)
                rng_state, row, col = get_stroke_coords_rolling(rng_state, r_s, c_s, r_e, c_e)
                rng_state = np.uint64(rng_state)
            else:
                row, col = cls.get_stroke_params(i, h, w, size, quadrant_input=quadrant_inputs[channel_idx], quadrant_padding=quadrant_padding)

            # APPLIES STROKE (WITH ITS 4 SLICES)
            if use_numba:
                stroke_indices, canvas_layer = stroke_numba(target_layer, canvas_layer, h, w, row, col, size, mult_arr)
            else:
                half = size // 2
                # Slices are [Row, Col]
                slices = [
                    (slice(row, row+half), slice(col, col+half)),
                    (slice(row, row+half), slice(col+half, col+size)),
                    (slice(row+half, row+size), slice(col, col+half)),
                    (slice(row+half, row+size), slice(col+half, col+size))
                ]
                
                stroke_indices = []
                for sl in slices:
                    # Basic boundary check (Never supposed to happen, but just in case)
                    if sl[0].start >= h or sl[1].start >= w:
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nSlice start out of bounds: {sl}, image size: ({h}, {w})\n----------\n\n\n")
                        continue

                    if sl[0].start >= sl[0].stop or sl[1].start >= sl[1].stop:
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nInvalid slice with start >= stop: {sl}\n----------\n\n\n")
                        continue
                    
                    # Extract region
                    target_slice = target_layer[sl]
                    canvas_slice = canvas_layer[sl]

                    # Another safety check for empty slices (which should not happen)
                    if target_slice.size == 0: 
                        stroke_indices.append(0)
                        print(f"\n\n\n----------\nEmpty slice encountered: {sl}\n----------\n\n\n")
                        continue

                    diff = target_slice - canvas_slice
                    mean_diff = np.mean(diff)
                    
                    best_idx = np.argmin(np.abs(mult_arr - mean_diff))
                    best_mult = mult_arr[best_idx]
                    
                    canvas_layer[sl] = np.clip(canvas_layer[sl] + best_mult, 0, 255)
                    stroke_indices.append(best_idx)

            # (MAIN LOOP BITS) stroke indices bits
            for idx in stroke_indices:
                bitstream += cls.encode_int(idx, signed=False, bitcount=multlist_bitcount)

            encoded_strokes.append(stroke_indices)
            quadrant_switch_counters[channel_idx] -= 1
            if channel_cycle:
                channel_cycle_timer -= 1

        return bitstream
    
    @classmethod
    def just_compress_file(cls, img_path, save_path=-1, **compress_kwargs):
        """ 
        Function to just take an image file path and return the compressed bitstream
        without collecting additional info or constructing the image.
        """
        img_pil = Image.open(img_path)
        if save_path is -1:
            save_path = f"compressed.pbc"
        bitstream = cls.just_compress(img_pil, **compress_kwargs)
        # BINARY SAVE
        try:
            with open(save_path, "wb") as f:
                b_data, pad_len = cls._bits_to_bytes(bitstream)
                f.write(bytes([pad_len])) 
                f.write(b_data)
            print(f"Binary bitstream saved to {save_path}")
        except Exception as e:
            print(f"Error saving bitstream: {e}")
        return bitstream

    @classmethod
    def simple_demo(cls, img, **compress_kwargs):
        """
        Simple demo function to compress an image and display results.
        """
        compressed_img, _, bitstream, losses = cls.compress(img, **compress_kwargs)
        original_size = img.size[0]*img.size[1]*3*8
        compressed_size = len(bitstream)
        compression_rate = original_size / compressed_size
        final_mse = int((losses[0] + losses[1] + losses[2]) / 3)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(compressed_img)
        plt.axis('off')
        plt.title(f'Compressed Image | {compressed_size / 8 / 1024:.2f} KB')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Original Image | {original_size / 8 / 1024:.2f} KB')
        plt.suptitle(f'Final MSE: {final_mse}\nCompression rate: {compression_rate:.2f}x', fontsize=16)
        plt.tight_layout()
        plt.show()

    @classmethod
    def simple_demo_from_file(cls, img_path, **compress_kwargs):
        """
        Simple demo function to compress an image from file and display results.
        """
        img_pil = Image.open(img_path)
        cls.simple_demo(img_pil, **compress_kwargs)

    @classmethod
    def simple_demo_vs_jpeg(cls, img, jpeg_quality=1, compare_efficiency=True, chart_jpeg_efficiency=False, **compress_kwargs):
        """
        Simple demo function to compress an image and display results. Compares PBC to JPEG at 1% quality.
        """
        pbc_timer = time.time()
        compressed_img, _, bitstream, losses = cls.compress(img, **compress_kwargs)
        pbc_timer = time.time() - pbc_timer

        jpeg_timer = time.time()
        jpeg_bitstream = cv2.imencode('.jpg', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])[1]
        jpeg_timer = time.time() - jpeg_timer

        original_size = img.size[0]*img.size[1]*3*8
        compressed_size = len(bitstream)
        compression_rate = original_size / compressed_size
        final_mse = int((losses[0] + losses[1] + losses[2]) / 3)

        jpeg_img = cv2.cvtColor(cv2.imdecode(jpeg_bitstream, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        jpeg_size = len(jpeg_bitstream)*8 # in bits
        jpeg_mse = int(np.mean((np.array(img).astype(np.float32) - np.array(jpeg_img).astype(np.float32)) ** 2))

        # compare mse / compression rate, calculate pbc efficiency vs jpeg, exact value not super important, just relative comparison
        def efficiency_formula(rate, mse, time):
            if mse == 0:
                return float('inf') # perfect efficiency for lossless compression
            return (rate / mse)# / time # commented out time factor for now to focus on size vs mse

        pbc_efficiency = efficiency_formula(compression_rate, final_mse, pbc_timer)
        jpeg_efficiency = efficiency_formula(original_size / jpeg_size, jpeg_mse, jpeg_timer)
        
        plt.figure(figsize=(18, 6), dpi=300)
        plt.subplot(1, 3, 1)
        plt.imshow(compressed_img)
        plt.axis('off')
        plt.title(f'PBC V2.3 | {compressed_size / 8 / 1024:.2f} KB\nCompression rate: {compression_rate:.2f}x | MSE: {final_mse}\nTime: {pbc_timer:.2f} s')
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Original Image | {original_size / 8 / 1024:.2f} KB')
        plt.subplot(1, 3, 3)
        plt.imshow(jpeg_img)
        plt.axis('off')
        plt.title(f'JPEG (Q:{jpeg_quality}%) | {jpeg_size / 8 / 1024:.2f} KB\nCompression rate: {original_size / jpeg_size:.2f}x | MSE: {jpeg_mse}\nTime: {jpeg_timer:.2f} s')
        if compare_efficiency:
            toptitle = f"PBC Efficiency Score: {pbc_efficiency:.2f} vs JPEG Efficiency Score: {jpeg_efficiency:.2f}"
            if pbc_efficiency > jpeg_efficiency:
                toptitle += "\nPBC wins!"
            else:
                toptitle += "\nJPEG wins!"
        else:
            toptitle = "PBC vs JPEG Comparison"
        plt.suptitle(toptitle, fontsize=16)
        plt.tight_layout()
        plt.show()

        if chart_jpeg_efficiency:
            qualities = list(range(1, 101, 9))
            jpeg_sizes = []
            jpeg_mses = []
            jpeg_efficiencies = []
            pbc_efficiencies = []

            plt.figure(figsize=(12, 6))

            for q in qualities:
                jpeg_timer = time.time()
                jpeg_bitstream = cv2.imencode('.jpg', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
                jpeg_timer = time.time() - jpeg_timer

                jpeg_size = len(jpeg_bitstream)*8 # in bits
                jpeg_img = cv2.cvtColor(cv2.imdecode(jpeg_bitstream, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                jpeg_mse = int(np.mean((np.array(img).astype(np.float32) - np.array(jpeg_img).astype(np.float32)) ** 2))
                
                jpeg_eff = efficiency_formula(original_size / jpeg_size, jpeg_mse, jpeg_timer)

                jpeg_sizes.append(jpeg_size / 8 / 1024) # in KB
                jpeg_mses.append(jpeg_mse)
                pbc_efficiencies.append(pbc_efficiency) # constant line for PBC

                if jpeg_mse == 0:
                    m = max(jpeg_efficiencies)
                    jpeg_efficiencies.append(m) 
                    plt.scatter([q], [m], label="JPEG Efficiency (Lossless, inf)", color="purple", marker='*')
                else:
                    jpeg_efficiencies.append(jpeg_eff)

            plt.plot(qualities, jpeg_efficiencies, label="JPEG Efficiency", color="blue")
            plt.plot(qualities, pbc_efficiencies, label="PBC Efficiency", color="orange", linestyle="--")
            plt.xlabel("JPEG Quality Setting (%)")
            plt.ylabel("Efficiency Score")
            plt.title("JPEG vs PBC Efficiency Comparison\n(PBC on default settings, JPEG varies by quality setting)")
            plt.legend()
            plt.show()

    @classmethod
    def simple_demo_vs_jpeg_from_file(cls, img_path, **compress_kwargs):
        """
        Simple demo function to compress an image from file and display results.
        """
        img_pil = Image.open(img_path)
        cls.simple_demo_vs_jpeg(img_pil, **compress_kwargs)

    @classmethod
    def simple_demo_vs_avif(cls, img, avif_quality=0, compare_efficiency=True, chart_avif_efficiency=False, **compress_kwargs):
        """
        Simple demo function to compress an image and display results. Compares PBC to AVIF at 0% quality.
        """
        pbc_timer = time.time()
        compressed_img, _, bitstream, losses = cls.compress(img, **compress_kwargs)
        pbc_timer = time.time() - pbc_timer

        avif_timer = time.time()
        img.save("temp_avif.avif", format="AVIF", quality=avif_quality)
        avif_timer = time.time() - avif_timer

        original_size = img.size[0]*img.size[1]*3*8
        compressed_size = len(bitstream)
        compression_rate = original_size / compressed_size
        final_mse = int((losses[0] + losses[1] + losses[2]) / 3)

        avif_img = Image.open("temp_avif.avif")
        avif_size = os.path.getsize("temp_avif.avif") * 8 # in bits
        avif_mse = int(np.mean((np.array(img).astype(np.float32) - np.array(avif_img).astype(np.float32)) ** 2))
        avif_compression_rate = original_size / avif_size

        # compare mse / compression rate, calculate pbc efficiency vs avif, exact value not super important, just relative comparison
        def efficiency_formula(rate, mse, time):
            if mse == 0:
                return float('inf') # perfect efficiency for lossless compression
            return (rate / mse)# / time # commented out time factor for now to focus on size vs mse

        pbc_efficiency = efficiency_formula(compression_rate, final_mse, pbc_timer)
        avif_efficiency = efficiency_formula(avif_compression_rate, avif_mse, avif_timer)
        
        plt.figure(figsize=(18, 6), dpi=300)
        plt.subplot(1, 3, 1)
        plt.imshow(compressed_img)
        plt.axis('off')
        plt.title(f'PBC V2.3 | {compressed_size / 8 / 1024:.2f} KB\nCompression rate: {compression_rate:.2f}x | MSE: {final_mse}\nTime: {pbc_timer:.2f} s')
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Original Image | {original_size / 8 / 1024:.2f} KB')
        plt.subplot(1, 3, 3)
        plt.imshow(avif_img)
        plt.axis('off')
        plt.title(f'AVIF (Q:{avif_quality}%) | {avif_size / 8 / 1024:.2f} KB\nCompression rate: {avif_compression_rate:.2f}x | MSE: {avif_mse}\nTime: {avif_timer:.2f} s')
        if compare_efficiency:
            toptitle = f"PBC Efficiency Score: {pbc_efficiency:.2f} vs AVIF Efficiency Score: {avif_efficiency:.2f}"
            if pbc_efficiency > avif_efficiency:
                toptitle += "\nPBC wins!"
            else:
                toptitle += "\nAVIF wins!"
        else:
            toptitle = "PBC vs AVIF Comparison"
        plt.suptitle(toptitle, fontsize=16)
        plt.tight_layout()
        plt.show()

        if chart_avif_efficiency:
            qualities = list(range(1, 101, 9))
            avif_sizes = []
            avif_mses = []
            avif_efficiencies = []
            pbc_efficiencies = []

            plt.figure(figsize=(12, 6))

            for q in qualities:
                avif_timer = time.time()
                img.save("temp_avif.avif", format="AVIF", quality=q)
                avif_timer = time.time() - avif_timer

                avif_size = os.path.getsize("temp_avif.avif") * 8 # in bits
                avif_img = Image.open("temp_avif.avif")
                avif_mse = int(np.mean((np.array(img).astype(np.float32) - np.array(avif_img).astype(np.float32)) ** 2))
                
                avif_eff = efficiency_formula(original_size / avif_size, avif_mse, avif_timer)

                avif_sizes.append(avif_size / 8 / 1024) # in KB
                avif_mses.append(avif_mse)
                pbc_efficiencies.append(pbc_efficiency) # constant line for PBC

                if avif_mse == 0:
                    m = max(avif_efficiencies)
                    avif_efficiencies.append(m) 
                    plt.scatter([q], [m], label="AVIF Efficiency (Lossless, inf)", color="purple", marker='*')
                else:
                    avif_efficiencies.append(avif_eff)

            plt.plot(qualities, avif_efficiencies, label="AVIF Efficiency", color="blue")
            plt.plot(qualities, pbc_efficiencies, label="PBC Efficiency", color="orange", linestyle="--")
            plt.xlabel("AVIF Quality Setting (%)")
            plt.ylabel("Efficiency Score")
            plt.title("AVIF vs PBC Efficiency Comparison\n(PBC on default settings, AVIF varies by quality setting)")
            plt.legend()
            plt.show()

    @classmethod
    def simple_demo_vs_avif_from_file(cls, img_path, **compress_kwargs):
        """
        Simple demo function to compress an image from file and display results.
        """
        img_pil = Image.open(img_path)
        cls.simple_demo_vs_avif(img_pil, **compress_kwargs)