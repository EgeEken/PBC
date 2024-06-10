import numpy as np
from PIL import Image
import time

def get_xy_size_from_seed(height, width, seed, sizerange):
    np.random.seed(seed)
    size = np.random.randint(sizerange[0], sizerange[1])
    x = np.random.randint(0, height - size)
    y = np.random.randint(0, width - size)
    return x, y, size


def get_slices_from_seed(matrix, seed, sizerange):
    x, y, size = get_xy_size_from_seed(matrix.shape[0], matrix.shape[1], seed, sizerange)
    
    xmid = x + size//2
    ymid = y + size//2
    
    xend = x + size
    yend = y + size
    
    return [
            matrix[x:xmid, y:ymid],
            matrix[x:xmid, ymid:yend], 
            matrix[xmid:xend, y:ymid], 
            matrix[xmid:xend, ymid:yend]
            ]
    
    
def get_diff_slices_from_seed(imgmatrix, matrix, seed, sizerange):
    x, y, size = get_xy_size_from_seed(matrix.shape[0], matrix.shape[1], seed, sizerange)
    
    xmid = x + size//2
    ymid = y + size//2
    
    xend = x + size
    yend = y + size
    
    return [
            imgmatrix[x:xmid, y:ymid].astype(int) - matrix[x:xmid, y:ymid] ,
            imgmatrix[x:xmid, ymid:yend].astype(int) - matrix[x:xmid, ymid:yend], 
            imgmatrix[xmid:xend, y:ymid].astype(int) - matrix[xmid:xend, y:ymid], 
            imgmatrix[xmid:xend, ymid:yend].astype(int) - matrix[xmid:xend, ymid:yend]
            ]


def best_i_list(imgmatrix, matrix, seed, sizerange, multlist):
    
    x, y, size = get_xy_size_from_seed(matrix.shape[0], matrix.shape[1], seed, sizerange)
    
    xmid = x + size//2
    ymid = y + size//2
    
    xend = x + size
    yend = y + size
    
    diff_slices = [
            imgmatrix[x:xmid, y:ymid].astype(int) - matrix[x:xmid, y:ymid] ,
            imgmatrix[x:xmid, ymid:yend].astype(int) - matrix[x:xmid, ymid:yend], 
            imgmatrix[xmid:xend, y:ymid].astype(int) - matrix[xmid:xend, y:ymid], 
            imgmatrix[xmid:xend, ymid:yend].astype(int) - matrix[xmid:xend, ymid:yend]
            ]
    
    i_list = [
        np.argmin(np.abs(multlist - np.mean(diff_slices[0]))),
        np.argmin(np.abs(multlist - np.mean(diff_slices[1]))),
        np.argmin(np.abs(multlist - np.mean(diff_slices[2]))),
        np.argmin(np.abs(multlist - np.mean(diff_slices[3])))
    ]
    
    matrix[x:xmid, y:ymid] = np.clip(matrix[x:xmid, y:ymid] + multlist[i_list[0]], 0, 255)
    matrix[x:xmid, ymid:yend] = np.clip(matrix[x:xmid, ymid:yend] + multlist[i_list[1]], 0, 255)
    matrix[xmid:xend, y:ymid] = np.clip(matrix[xmid:xend, y:ymid] + multlist[i_list[2]], 0, 255)
    matrix[xmid:xend, ymid:yend] = np.clip(matrix[xmid:xend, ymid:yend] + multlist[i_list[3]], 0, 255)
    
    return i_list
    
    
def apply_brush(matrix, seed, sizerange, multlist, i_list):
    x, y, size = get_xy_size_from_seed(matrix.shape[0], matrix.shape[1], seed, sizerange)
    
    xmid = x + size//2
    ymid = y + size//2
    
    xend = x + size
    yend = y + size
    
    matrix[x:xmid, y:ymid] = np.clip(matrix[x:xmid, y:ymid] + multlist[i_list[0]], 0, 255)
    matrix[x:xmid, ymid:yend] = np.clip(matrix[x:xmid, ymid:yend] + multlist[i_list[1]], 0, 255)
    matrix[xmid:xend, y:ymid] = np.clip(matrix[xmid:xend, y:ymid] + multlist[i_list[2]], 0, 255)
    matrix[xmid:xend, ymid:yend] = np.clip(matrix[xmid:xend, ymid:yend] + multlist[i_list[3]], 0, 255)
    
    
def compress(img, sizerange, multlist, stroke_count=5000):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    startcolors = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
    
    testR = np.full_like(r, startcolors[0], dtype=int)
    testG = np.full_like(g, startcolors[1], dtype=int)
    testB = np.full_like(b, startcolors[2], dtype=int)

    res = []

    for _ in range(stroke_count):
        if _ % 3 == 0:
            temp = best_i_list(r, testR, _, sizerange, multlist)
            res.append(temp)
        elif _ % 3 == 1:
            temp = best_i_list(g, testG, _, sizerange, multlist)
            res.append(temp)
        else:
            temp = best_i_list(b, testB, _, sizerange, multlist)
            res.append(temp)

    return np.stack([testR, testG, testB], axis=-1).astype(np.uint8), res, startcolors

def decompress(height, width, startcolors, sizerange, multlist, strokes):
    r = np.full((height, width), startcolors[0], dtype=int)
    g = np.full((height, width), startcolors[1], dtype=int)
    b = np.full((height, width), startcolors[2], dtype=int)
    
    for i in range(len(strokes)):
        if i % 3 == 0:
            apply_brush(r, i, sizerange, multlist, strokes[i])
        elif i % 3 == 1:
            apply_brush(g, i, sizerange, multlist, strokes[i])
        else:
            apply_brush(b, i, sizerange, multlist, strokes[i])
            
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def encode_strokes(strokes, multlist):
    bits_per_quadrant = int(np.ceil(np.log2(len(multlist))))
    res = ""
    for stroke in strokes:
        for i in stroke:
            res += bin(i)[2:].zfill(bits_per_quadrant)
    return res

def decode_strokes(encoded, multlist):
    res = []
    bits_per_quadrant = int(np.ceil(np.log2(len(multlist))))
    for i in range(0, len(encoded), bits_per_quadrant*4):
        temp = []
        for j in range(i, i+bits_per_quadrant*4, bits_per_quadrant):
            temp.append(int(encoded[j:j+bits_per_quadrant], 2))
        res.append(temp)
    return res

def encode_header(height, width, startcolors, sizerange, multlist):
    res = ""
    res += bin(height)[2:].zfill(16)
    res += bin(width)[2:].zfill(16)
    res += bin(startcolors[0])[2:].zfill(8)
    res += bin(startcolors[1])[2:].zfill(8)
    res += bin(startcolors[2])[2:].zfill(8)
    res += bin(sizerange[0])[2:].zfill(16)
    res += bin(sizerange[1])[2:].zfill(16)
    
    for mult in multlist+256:
        res += bin(mult)[2:].zfill(9)

    return res + "000000000" # 9 bits padding for the header

def decode_header(encoded):
    height = int(encoded[:16], 2)
    width = int(encoded[16:32], 2)
    startcolors = (int(encoded[32:40], 2), int(encoded[40:48], 2), int(encoded[48:56], 2))
    sizerange = (int(encoded[56:72], 2), int(encoded[72:88], 2))
    
    multlist = []
    i = 88
    while encoded[i:i+9] != "000000000":
        multlist.append(int(encoded[i:i+9], 2)-256)
        i += 9
    return height, width, startcolors, sizerange, np.array(multlist)

def encode_all(height, width, startcolors, sizerange, multlist, strokes):
    encoded_strokes = encode_strokes(strokes, multlist)
    header = encode_header(height, width, startcolors, sizerange, multlist)
    return header + encoded_strokes

def decode_all(encoded):
    height, width, startcolors, sizerange, multlist = decode_header(encoded)
    strokes = decode_strokes(encoded[88 + len(multlist)*9 + 9:], multlist)
    return (height, width, startcolors, sizerange, multlist, strokes)


def save_encoded_file(filename, encoded):
    with open(filename + ".bin", "wb") as f:
        f.write(int("1" + encoded, 2).to_bytes((len("1" + encoded) + 7) // 8, byteorder="big"))
        
def read_encoded_file(filename):
    with open(filename + ".bin", "rb") as f:
        read_encoded = bin(int.from_bytes(f.read(), byteorder="big"))[3:]
    return read_encoded

def compress_into_file(filename, img, sizerange, multlist, stroke_count=10000):
    compressed = compress(img, sizerange, multlist, stroke_count)
    encoded = encode_all(img.shape[0], img.shape[1], compressed[2], sizerange, multlist, compressed[1])
    save_encoded_file(filename, encoded)
    
def recreate_from_file(filename):
    return decompress(*decode_all(read_encoded_file(filename)))