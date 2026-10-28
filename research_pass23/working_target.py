import numpy as np
from PIL import Image
from PBC3 import PBC3

RGB_TO_YCBCR=np.array([[.299,.587,.114],[-.168736,-.331264,.5],[.5,-.418688,-.081312]],dtype=np.float32)


def reconstruct(target,prep):
    output=PBC3._canvas_to_image(target,prep['color_space'],False)
    size=(prep['original_w'],prep['original_h'])
    if prep['downsampled'] and output.size!=size:
        output=output.resize(size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    return np.asarray(output,dtype=np.uint8)


def adjust(prep,step,iterations,limit,diagnostics=None):
    if prep['has_alpha'] or prep['warm_target'] is not None or prep['color_space'] not in ('YCbCr','RGB'):
        raise ValueError('research target requires RGB input without alpha or warmup')
    ordinary=prep['target']
    if not iterations or not prep['downsampled']:return ordinary.copy()
    source=np.asarray(prep['orig_compare'],dtype=np.uint8)
    working=ordinary.astype(np.float32)
    for _ in range(iterations):
        canvas=np.clip(np.rint(working),0,255).astype(np.int32)
        predicted=reconstruct(canvas,prep)
        correction=np.empty(ordinary.shape,dtype=np.float32)
        for c in range(3):
            error=source[:,:,c].astype(np.float32)-predicted[:,:,c].astype(np.float32)
            layer=Image.fromarray(error)
            correction[:,:,c]=np.asarray(layer.resize((prep['w'],prep['h']),PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP),dtype=np.float32)
        if prep['color_space']=='YCbCr':correction=correction @ RGB_TO_YCBCR.T
        proposed=working+np.float32(step)*correction
        bounded=np.clip(proposed,ordinary-limit,ordinary+limit)
        if diagnostics is not None:
            diagnostics.append(dict(outside_color_range=int(np.count_nonzero((bounded<0)|(bounded>255))),deviation_cap_hits=int(np.count_nonzero(proposed!=bounded)),max_projected_correction=float(np.max(np.abs(correction))),min_before_color_clip=float(bounded.min()),max_before_color_clip=float(bounded.max())))
        working=np.clip(bounded,0,255).astype(np.float32)
    return np.clip(np.rint(working),0,255).astype(np.int32)
