import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass41'
OUT = PASS / 'outputs'
LEDGER = json.loads((OUT / 'phase_b_results.json').read_text())
FFMPEG = pathlib.Path(r'C:\ffmpeg\bin\ffmpeg.exe')

def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)

def mse(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return float(np.mean(d*d))

def channel_mse(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return [float(np.mean(d[:,:,i]**2)) for i in range(3)]

def edge_mask(rgb):
    v=rgb.astype(np.float64)/255.0
    lin=np.where(v<=0.04045,v/12.92,((v+0.055)/1.055)**2.4)
    y=lin@np.array([0.2126,0.7152,0.0722])
    grad=np.hypot(sobel(y,axis=1),sobel(y,axis=0))
    return grad>=np.quantile(grad,0.8)

results=[]
for row in LEDGER['rows']:
    if row['phase']!='phaseB_timed' or row['repeat']!=3:
        continue
    image,q=row['image'],row['qcolor']
    stem=pathlib.Path(image).stem
    prefix=OUT/'planes'/f'{stem}_q{q}'
    meta={k:v for k,v in (line.strip().split('=',1) for line in pathlib.Path(str(prefix)+'.meta').read_text().splitlines())}
    width,height,depth=(int(meta[k]) for k in ('width','height','depth'))
    source=read_rgb(ROOT/'hpt_data'/image)
    if source.shape[:2]!=(height,width) or depth!=8 or int(meta['yuv_format'])!=3 or int(meta['yuv_range'])!=1:
        raise RuntimeError(f'unexpected decoded AVIF properties: {meta}')
    # AVIF_PIXEL_FORMAT_YUV420 is 3; chroma plane dimensions are ceil-divided.
    cw,ch=(width+1)//2,(height+1)//2
    planes={}
    for label,shape in [('Y',(height,width)),('U',(ch,cw)),('V',(ch,cw))]:
        a=np.fromfile(str(prefix)+f'_ref_{label}.bin',dtype=np.uint8).reshape(shape)
        b=np.fromfile(str(prefix)+f'_decoded_{label}.bin',dtype=np.uint8).reshape(shape)
        planes[label]={'reference_mean':float(a.mean()),'decoded_mean':float(b.mean()),
                       'reference_std':float(a.std()),'decoded_std':float(b.std()),
                       'reference_min':int(a.min()),'reference_max':int(a.max()),
                       'decoded_min':int(b.min()),'decoded_max':int(b.max()),'mse':mse(a,b),
                       'sse':float(np.sum((a.astype(np.float64)-b.astype(np.float64))**2))}
    actual=read_rgb(OUT/'benchmark'/f's6_{stem}_q{q}_A_phaseB_timed3.png')
    custom=np.fromfile(str(prefix)+'_decoded_rgb.bin',dtype=np.uint8).reshape((height,width,3))
    source_yuv_rgb=np.fromfile(str(prefix)+'_reference_rgb.bin',dtype=np.uint8).reshape((height,width,3))
    decy_refuv=np.fromfile(str(prefix)+'_decodedY_referenceUV_rgb.bin',dtype=np.uint8).reshape((height,width,3))
    refy_decyuv=np.fromfile(str(prefix)+'_referenceY_decodedUV_rgb.bin',dtype=np.uint8).reshape((height,width,3))
    custom_diff=int(np.max(np.abs(custom.astype(np.int16)-actual.astype(np.int16))))
    if custom_diff>0:
        raise RuntimeError(f'libavif custom reconstruction differs from avifdec PNG by {custom_diff} levels: {image}')
    edge=edge_mask(source); smooth=~edge
    actual_mse=mse(source,actual)
    cf_luma=mse(source,decy_refuv)
    cf_chroma=mse(source,refy_decyuv)
    ref_floor=mse(source,source_yuv_rgb)
    info=subprocess.run([str(ROOT/'research_pass35'/'outputs'/'binaries'/'reference'/'avifdec.exe'),'--info',str(OUT/'benchmark'/f's6_{stem}_q{q}_A_phaseB_timed3.avif')],capture_output=True,text=True,check=True)
    (OUT/'planes'/f'{stem}_q{q}_avif_info.txt').write_text(info.stdout)
    result={'image':image,'qcolor':q,'source_sha256':row['source_sha256'],'avif_sha256':row['avif_sha256'],
            'bytes':row['bytes'],'bpp':row['bpp'],'actual_rgb_mse':actual_mse,'actual_rgb_psnr':float(10*np.log10(255**2/actual_mse)),
            'actual_rgb_channel_mse':channel_mse(source,actual),'edge_rgb_mse':mse(source[edge],actual[edge]),
            'smooth_rgb_mse':mse(source[smooth],actual[smooth]),'edge_fraction':float(edge.mean()),
            'source_yuv_roundtrip_rgb_mse_floor':ref_floor,'decoded_rgb_custom_vs_avifdec_max_abs':custom_diff,
            'counterfactual_decodedY_referenceUV_rgb_mse':cf_luma,
            'counterfactual_referenceY_decodedUV_rgb_mse':cf_chroma,
            'rgb_mse_change_if_only_chroma_restored':actual_mse-cf_luma,
            'rgb_mse_change_if_only_luma_restored':actual_mse-cf_chroma,
            'counterfactuals_non_additive':True,'native_planes':planes,
            'avif_properties':{k:int(meta[k]) for k in ('width','height','depth','yuv_format','yuv_range','chroma_position','primaries','transfer','matrix')},
            'encode_seconds':[r['encode_seconds'] for r in LEDGER['rows'] if r['image']==image and r['qcolor']==q and r['phase']=='phaseB_timed']}
    results.append(result)
    print(f"{image:14} RGB MSE {actual_mse:.4f}; Y/U/V MSE {planes['Y']['mse']:.3f}/{planes['U']['mse']:.3f}/{planes['V']['mse']:.3f}; restore UV {actual_mse-cf_luma:+.4f}; restore Y {actual_mse-cf_chroma:+.4f}; edge/smooth {result['edge_rgb_mse']:.4f}/{result['smooth_rgb_mse']:.4f}; roundtrip max {custom_diff}",flush=True)
result={'method':{'source_yuv':'Pinned libavif RGB-to-YUV conversion on EXIF-oriented RGB with YUV420, full range, CICP 1/13/1, automatic chroma downsampling, no SharpYUV.','decoded_yuv':'Native Y/U/V planes from pinned libavif decode of the measured AVIF.','counterfactual_rgb':'Pinned libavif YUV-to-RGB with AVIF signaled range, CICP, chroma position, and automatic chroma upsampling.','counterfactual_warning':'Replacing Y or U/V is diagnostic; RGB MSE changes are not additive because conversion, clipping and channel interactions remain.'},'results':results}
(OUT/'chroma_error_analysis.json').write_text(json.dumps(result,indent=2)+'\n')

