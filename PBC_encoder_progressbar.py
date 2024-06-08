

# |------------------------------------------------------------------------------|
# | This is the V1.0 Encoder for the Probabilistic Brush Compression algorithm.  |
# | Developed by: Ege Eken, 2024, github.com/EgeEken/PBC                         |
# |------------------------------------------------------------------------------|


import numpy as np
from PIL import Image
import time
import itertools

import progressbar as pb

def get_pos_from_seed(matrix, seed):
    np.random.seed(seed)
    return np.random.randint(0, matrix.shape[0]), np.random.randint(0, matrix.shape[1])

def brush(matrix, x, y, attributes):
    """Generates a brush stroke layer with the given position, size and multiplier

    x, y: positions
    size: int ( 1 - 255 )
    multiplier: int ( (-255) - 255 )
    """
    size = attributes[0]
    multiplier = attributes[1]
    # add more attributes in the future
    
    res = np.zeros(matrix.shape)
    
    x_min = max(x - size, 0)
    x_max = min(x + size + 1, res.shape[0])
    y_min = max(y - size, 0)
    y_max = min(y + size + 1, res.shape[1])
    
    res[x_min:x_max, y_min:y_max] = multiplier
    
    return res

def brush_check(imgmatrix, matrix, x, y, attributes):
    """ 
    Negative improvement means the brush made the section worse, positive means it got better 
    The higher the better the lower the worse
    """
    size = attributes[0]
    multiplier = attributes[1]
    # add more attributes in the future
    
    improvements = []
    res = np.zeros(matrix.shape)
    
    x_min = max(x - size, 0)
    x_max = min(x + size + 1, matrix.shape[0])
    y_min = max(y - size, 0)
    y_max = min(y + size + 1, matrix.shape[1])
    
    res[x_min:x_max, y_min:y_max] = multiplier
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            original_diff = abs(imgmatrix[i, j] - matrix[i, j])
            new_diff = abs(imgmatrix[i, j] - np.clip(matrix[i, j] + multiplier, 0, 255))
            improvements.append(original_diff - new_diff)
            
    return res, improvements

def best_brush(imgmatrix, matrix, x, y, attributes_list):
    best_attributes = None
    best_improvement = 0
    best_stroke = None
    # check every combination of size and multiplier (and more attributes in the future) in attributes_list
    index_ranges = [range(len(attr_list)) for attr_list in attributes_list]

    for indices in itertools.product(*index_ranges):
        # Get the actual attribute values using the indices
        attributes = [attributes_list[i][index] for i, index in enumerate(indices)]
        stroke, improvements = brush_check(imgmatrix, matrix, x, y, attributes)
        
        temp = np.sum(improvements)
        if temp > best_improvement:
            best_improvement = temp
            best_attributes = indices
            best_stroke = stroke
    
    return best_stroke, best_attributes, best_improvement


def RGB_find_strokes(imgmatrix, attributes_list, startcolors=(128,128,128), max_seed=300, max_seed_count=100):
    r, g, b = imgmatrix[:,:,0], imgmatrix[:,:,1], imgmatrix[:,:,2]
    Rs, Gs, Bs = startcolors
    
    Rmatrix = np.full_like(r, Rs, dtype=int)
    Gmatrix = np.full_like(g, Gs, dtype=int)
    Bmatrix = np.full_like(b, Bs, dtype=int)
    
    Rres = []
    Gres = []
    Bres = []
    
    seed = 0
    addedseed = 0
    
    with pb.ProgressBar(max_value=max_seed) as bar:
        while seed < max_seed and addedseed < max_seed_count:
            seed += 1
            bar.update(seed)

            if seed % 3 == 0: # Red
                x, y = get_pos_from_seed(Rmatrix, seed)
                best_stroke, best_attributes, best_improvement = best_brush(r, Rmatrix, x, y, attributes_list)
                if best_improvement <= 0:
                    continue
                Rmatrix = np.clip(Rmatrix + best_stroke, 0, 255)
                Rres.append((seed, best_attributes))
                addedseed += 1

            elif seed % 3 == 1: # Green
                x, y = get_pos_from_seed(Gmatrix, seed)
                best_stroke, best_attributes, best_improvement = best_brush(g, Gmatrix, x, y, attributes_list)
                if best_improvement <= 0:
                    continue
                Gmatrix = np.clip(Gmatrix + best_stroke, 0, 255)
                Gres.append((seed, best_attributes))
                addedseed += 1

            elif seed % 3 == 2: # Blue
                x, y = get_pos_from_seed(Bmatrix, seed)
                best_stroke, best_attributes, best_improvement = best_brush(b, Bmatrix, x, y, attributes_list)
                if best_improvement <= 0:
                    continue
                Bmatrix = np.clip(Bmatrix + best_stroke, 0, 255)
                Bres.append((seed, best_attributes))
                addedseed += 1

    return (Rmatrix, Gmatrix, Bmatrix), (Rres, Gres, Bres)

def RGB_find_more_strokes(imgmatrix, matrixes, strokes, attributes_list, max_seed=300, max_seed_count=100):
    r, g, b = imgmatrix[:,:,0], imgmatrix[:,:,1], imgmatrix[:,:,2]
    
    Rmatrix, Gmatrix, Bmatrix = matrixes[:,:,0], matrixes[:,:,1], matrixes[:,:,2]
    Rres, Gres, Bres = strokes
    
    startseed = np.max([np.max(strokes[i]) for i in range(3)]) + 1
    seed = 0
    addedseed = 0
    
    while seed < startseed + max_seed and addedseed < max_seed_count:
        seed += 1
        if seed % 3 == 0: # Red
            x, y = get_pos_from_seed(Rmatrix, seed)
            best = best_brush(r, Rmatrix, x, y, attributes_list)
            if best[-1] <= 0:
                continue
            Rmatrix = np.clip(Rmatrix + best[0], 0, 255)
            Rres.append((seed, best[1]))
            addedseed += 1
            
        elif seed % 3 == 1: # Green
            x, y = get_pos_from_seed(Gmatrix, seed)
            best = best_brush(g, Gmatrix, x, y, attributes_list)
            if best[-1] <= 0:
                continue
            Gmatrix = np.clip(Gmatrix + best[0], 0, 255)
            Gres.append((seed, best[1]))
            addedseed += 1
            
        elif seed % 3 == 2: # Blue
            x, y = get_pos_from_seed(Bmatrix, seed)
            best = best_brush(b, Bmatrix, x, y, attributes_list)
            if best[-1] <= 0:
                continue
            Bmatrix = np.clip(Bmatrix + best[0], 0, 255)
            Bres.append((seed, best[1]))
            addedseed += 1
            
    return (Rmatrix, Gmatrix, Bmatrix), (Rres, Gres, Bres)


def get_real_attributes(height, width, data_attributes_list):
    sizelist = np.ceil(min(height, width) / np.array(data_attributes_list[0])).astype(int)
    
    multlist = data_attributes_list[1]
    # the first half of the multlist is turned negative (* -1), the second half stays positive
    multlist = np.hstack((np.array(multlist[:len(multlist) // 2]) * -1, multlist[len(multlist) // 2:])).astype(int)
    
    # add more attributes in the future
    
    return [sizelist, multlist]

def compress_image(img, data_attributes_list, startcolors=(128,128,128), max_seed=300, max_seed_count=100, verbose=False):
    
    
    height, width = img.shape[0], img.shape[1]
    
    attributes_list = get_real_attributes(height, width, data_attributes_list)
    bits_per_attribute = [int(np.ceil(np.log2(len(att)))) for att in attributes_list]
    
    bit_per_stroke = 1
    for i in range(len(attributes_list)):
        bit_per_stroke += bits_per_attribute[i]
    
    if verbose:
        print(f"Bits per stroke: {bit_per_stroke}")
    timer_start = time.time()
    Paintings, Strokes = RGB_find_strokes(img, attributes_list, startcolors, max_seed, max_seed_count)
    if verbose:
        print(f"Encoding time: {time.time() - timer_start:.2f}s")
    
    Rpainting, Gpainting, Bpainting = Paintings
    Rstrokes, Gstrokes, Bstrokes = Strokes
    strokecount = len(Rstrokes) + len(Gstrokes) + len(Bstrokes)
    if verbose:
        print(f"Stroke count: {len(Rstrokes), len(Gstrokes), len(Bstrokes)} (total: {strokecount})")
    bitcount = bit_per_stroke * strokecount
    if verbose:
        print(f"Total data bits: {bitcount} bits / {bitcount // 8} bytes / {bitcount / 8 / 1024:.2f} KB")
        original_bitcount = height * width * 8 * 3
        print(f"Original uncompressed image size: {original_bitcount} bits / {original_bitcount // 8} bytes / {original_bitcount / 8 / 1024:.2f} KB")
        print(f"Compression ratio: {original_bitcount / bitcount:.2f}x / {bitcount * 100 / original_bitcount:.3f}%")
    return np.stack([Rpainting, Gpainting, Bpainting], axis=2).astype(np.uint8), Strokes

def compress_more(img, matrixes, strokes, data_attributes_list, max_seed=300, max_seed_count=100):
    
    attributes_list = get_real_attributes(img, data_attributes_list)
    bits_per_attribute = [int(np.ceil(np.log2(len(att)))) for att in attributes_list]
    
    bit_per_stroke = 1
    for i in range(len(attributes_list)):
        bit_per_stroke += bits_per_attribute[i]
    
    
    print(f"Bits per stroke: {bit_per_stroke}")
    height, width = img.shape[0], img.shape[1]
    timer_start = time.time()
    Paintings, Strokes = RGB_find_more_strokes(img, matrixes, strokes, attributes_list, max_seed, max_seed_count)
    print(f"Encoding time: {time.time() - timer_start:.2f}s")
    Rpainting, Gpainting, Bpainting = Paintings
    Rstrokes, Gstrokes, Bstrokes = Strokes
    strokecount = len(Rstrokes) + len(Gstrokes) + len(Bstrokes)
    print(f"Stroke count: {len(Rstrokes), len(Gstrokes), len(Bstrokes)} (total: {strokecount})")
    bitcount = bit_per_stroke * strokecount
    print(f"Total data bits: {bitcount} bits / {bitcount // 8} bytes / {bitcount / 8 / 1024:.2f} KB")
    original_bitcount = height * width * 8 * 3
    print(f"Original uncompressed image size: {original_bitcount} bits / {original_bitcount // 8} bytes / {original_bitcount / 8 / 1024:.2f} KB")
    print(f"Compression ratio: {original_bitcount / bitcount:.2f}x / {bitcount * 100 / original_bitcount:.3f}%")
    return np.stack([Rpainting, Gpainting, Bpainting], axis=2).astype(np.uint8), Strokes

def decompress_image(strokes, height, width, data_attributes_list, startcolors=(128,128,128)):
    
    attributes_list = get_real_attributes(height, width, data_attributes_list)

    Rs, Gs, Bs = startcolors
    Rstrokes, Gstrokes, Bstrokes = strokes
    
    Rmatrix = np.full((height, width), Rs, dtype=int)
    Gmatrix = np.full((height, width), Gs, dtype=int)
    Bmatrix = np.full((height, width), Bs, dtype=int)
    
    for stroke in Rstrokes:
        x, y = get_pos_from_seed(Rmatrix, stroke[0])
        Rmatrix = np.clip(Rmatrix + brush(Rmatrix, x, y, [attributes_list[i][stroke[1][i]] for i in range(len(attributes_list))]), 0, 255)
    for stroke in Gstrokes:
        x, y = get_pos_from_seed(Gmatrix, stroke[0])
        Gmatrix = np.clip(Gmatrix + brush(Gmatrix, x, y, [attributes_list[i][stroke[1][i]] for i in range(len(attributes_list))]), 0, 255)
    for stroke in Bstrokes:
        x, y = get_pos_from_seed(Bmatrix, stroke[0])
        Bmatrix = np.clip(Bmatrix + brush(Bmatrix, x, y, [attributes_list[i][stroke[1][i]] for i in range(len(attributes_list))]), 0, 255)
    return np.stack([Rmatrix, Gmatrix, Bmatrix], axis=2).astype(np.uint8)

def sort_strokes(strokes):
    return sorted(strokes[0] + strokes[1] + strokes[2])

def unsort_strokes(strokes):
    Rstrokes = []
    Gstrokes = []
    Bstrokes = []
    for stroke in strokes:
        if stroke[0] % 3 == 0:
            Rstrokes.append(stroke)
        elif stroke[0] % 3 == 1:
            Gstrokes.append(stroke)
        elif stroke[0] % 3 == 2:
            Bstrokes.append(stroke)
    return Rstrokes, Gstrokes, Bstrokes

def encode_strokes(strokes, attributes_list, space_bits=4):
    # sizelist = attributes_list[0]
    # multlist = attributes_list[1]
    # more attributes can be added, but code needs to be modified
    
    sorted_strokes = sort_strokes(strokes)
    bits_per_attribute = [int(np.ceil(np.log2(len(att)))) for att in attributes_list]
    
    bit_per_stroke = 1
    for i in range(len(attributes_list)):
        bit_per_stroke += bits_per_attribute[i]
        
    res = ""
    curr_seed = 0
    
    for stroke in sorted_strokes:
        if stroke[0] != curr_seed: # space between current seed and stroke seed
            while curr_seed < stroke[0]:
                space = stroke[0] - curr_seed
                res += "0"
                add_space = np.clip(space, 1, 2**space_bits)
                res += bin(add_space - 1)[2:].zfill(space_bits)
                curr_seed += add_space
            # spaces have been added, now onto the stroke
        res += "1"
        for att in range(len(attributes_list)):
            if bits_per_attribute[att] > 0:
                res += bin(stroke[1][att])[2:].zfill(bits_per_attribute[att])
        curr_seed += 1
    
    return res

def decode_strokes(encoded, attributes_list, space_bits=4):
    # sizelist = attributes_list[0]
    # multlist = attributes_list[1]
    # more attributes can be added, but code needs to be modified
    
    bits_per_attribute = [int(np.ceil(np.log2(len(att)))) for att in attributes_list]
    
    bits_per_stroke = 0
    for i in range(len(attributes_list)):
        bits_per_stroke += bits_per_attribute[i]
    
    strokes = []
    
    curr_seed = 0
    curr_atts = []
    
    i = 0
    
    while i < len(encoded):
        
        if encoded[i] == "0":
            # space byte
            i += 1
            curr_seed += int(encoded[i:i+space_bits], 2) + 1
            i += space_bits
            
        elif encoded[i] == "1":
            # stroke byte
            i += 1
            curr_atts = []
            for att in range(len(attributes_list)):
                if bits_per_attribute[att] > 0:
                    curr_atts.append(int(encoded[i:i+bits_per_attribute[att]]))
                    i += bits_per_attribute[att]
                else:
                    curr_atts.append(0)
            
            strokes.append((curr_seed, tuple(curr_atts)))
            curr_seed += 1
            
            
    return unsort_strokes(strokes)

def encode_header(height, width, data_attributes_list, startcolors=(128,128,128), space_bits=4):
    res = ""
    res += bin(height)[2:].zfill(13) # height
    res += bin(width)[2:].zfill(13) # width
    for sc in startcolors:
        res += bin(sc)[2:].zfill(8) # start colors
    res += bin(space_bits)[2:].zfill(6)  # space bit count
    for att_list in data_attributes_list:
        for att in att_list:
            res += bin(att)[2:].zfill(8)
        res += "00000000"
    return res

def decode_header(encoded_header):
    i = 0
    height = int(encoded_header[i:i+13], 2)
    i += 13
    width = int(encoded_header[i:i+13], 2)
    i += 13
    startcolors = []
    for _ in range(3):
        startcolors.append(int(encoded_header[i:i+8], 2))
        i += 8
    startcolors = tuple(startcolors)
    space_bit_count = int(encoded_header[i:i+6], 2)
    i += 6
    
    data_attributes_list = []
    att_bytes = [encoded_header[i:][j:j+8] for j in range(0, len(encoded_header[i:]), 8)]
    temp = []
    for a_b in att_bytes:
        if a_b == "00000000":
            data_attributes_list.append(temp)
            temp = []
        else:
            temp.append(int(a_b, 2))
    
    
    return height, width, startcolors, space_bit_count, data_attributes_list 


def complete_encode(img, strokes, data_attributes_list, startcolors=(128,128,128), space_bits=4):
    encoded_strokes = encode_strokes(strokes, data_attributes_list, space_bits)
    encoded_header = encode_header(img.shape[0], img.shape[1], data_attributes_list, startcolors, space_bits)
    return encoded_header + encoded_strokes

def complete_decode(complete_encoded):
    
    attribute_counter = 2 # current attribute list has 2 attributes, size and multiplier
    
    
    height = int(complete_encoded[:13], 2)
    width = int(complete_encoded[13:26], 2)
    startcolors = (int(complete_encoded[26:34], 2), int(complete_encoded[34:42], 2), int(complete_encoded[42:50], 2))
    space_bits = int(complete_encoded[50:56], 2)
    
    i = 56
    data_attributes_list = []
    att_bytes = [complete_encoded[56:][j:j+8] for j in range(0, len(complete_encoded[56:]), 8)]
    temp = []
    for a_b in att_bytes:
        i += 8
        if a_b == "00000000":
            data_attributes_list.append(temp)
            temp = []
            attribute_counter -= 1
            if attribute_counter == 0:
                break
        else:
            temp.append(int(a_b, 2))
            
    strokes = decode_strokes(complete_encoded[i:], data_attributes_list, space_bits)
    
    return strokes, height, width, data_attributes_list, startcolors

def save_encoded_file(filename, encoded):
    with open(filename + ".bin", "wb") as f:
        f.write(int("1" + encoded, 2).to_bytes((len("1" + encoded) + 7) // 8, byteorder="big"))

def read_encoded_file(filename):
    with open(filename + ".bin", "rb") as f:
        read_encoded = bin(int.from_bytes(f.read(), byteorder="big"))[3:]
    return read_encoded

def compress_into_file(filename, img, data_attributes_list, startcolors=(128,128,128), max_seed=300, max_seed_count=100, space_bits=4, verbose=False):
    timer_start = time.time()
    encoded = complete_encode(img, compress_image(img, data_attributes_list, startcolors, max_seed, max_seed_count)[1], data_attributes_list, startcolors, space_bits)
    save_encoded_file(filename, encoded)
    
    print(f"--------\nFile saved as {filename}.bin in {time.time() - timer_start:.2f}s")
    
    if verbose:
        bitcount = len(encoded)
        print(f"Total data bits: {bitcount} bits / {bitcount // 8} bytes / {bitcount / 8 / 1024:.2f} KB")
        original_bitcount = img.shape[0] * img.shape[1] * 8 * 3
        print(f"Original uncompressed image size: {original_bitcount} bits / {original_bitcount // 8} bytes / {original_bitcount / 8 / 1024:.2f} KB")
        print(f"Compression ratio: {original_bitcount / bitcount:.2f}x / {bitcount * 100 / original_bitcount:.3f}%")

def recreate_from_file(filename):
    return decompress_image(*complete_decode(read_encoded_file(filename)))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


def main():
    
    filename = input("Enter the image filename: ")
    img = np.asarray(Image.open(filename).convert("RGB"))
    
    attribute_list = [
        [20], # size
        [35, 30] # multiplier
        # future models will have more attribute types
    ]
    
    max_seed = int(input("Enter the maximum seed value (it will search up to this value): "))
    max_seed_count = int(input("Enter the maximum seed count (after finding this many seeds, it will stop): "))
    space_bit_count = 4
    
    
    compress_into_file(filename + "_encoded", img,
                       attribute_list,
                       (int(np.mean(img[:,:,0])), int(np.mean(img[:,:,1])), int(np.mean(img[:,:,2]))),
                       max_seed, max_seed_count, space_bit_count, True)
    
    input("Done! Use the decoder to recreate the image from the binary file. Press Enter to exit.")
    
    
if __name__ == "__main__":
    main()
