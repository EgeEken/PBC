from __future__ import annotations
import hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import numba
import numpy as np
from PIL import Image,ImageOps
from scipy.ndimage import sobel
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'research_pass33'));sys.path.insert(0,str(ROOT))
import pbc33_codec as p33
import pbc34_codec as p34
sys.path.insert(0,str(ROOT/'research_pass31'))
import pbc31_codec as p31
numba.set_num_threads(1)
LAMBDAS=(0.0,20.0,100.0);SPARSE_LAMBDAS=(400.0,1600.0);QSTEPS=(16,32,64);BANDS=('F','L','H');REPEATS=3

def sha(data): return hashlib.sha256(data).hexdigest()
def masks(source):
    gray=source.astype(np.float32).mean(2);mag=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1));flat=mag.ravel()
    return mag>=np.quantile(flat,.9),mag<=np.quantile(flat,.5)
def metrics(source,recon,edge,smooth):
    err=np.square(source.astype(np.int32)-recon.astype(np.int32),dtype=np.int64).sum(2)
    return {'mse':float(err.mean()/3),'edge_top10_mse':float(err[edge].mean()/3),'smooth_bottom50_mse':float(err[smooth].mean()/3)}
def p33_account(data):
    fields=p33.HEADER.unpack_from(data);_,_,w,h,_,_,_,q,_,n,rawlen,xz=fields
    decomp=__import__('lzma').LZMADecompressor(format=__import__('lzma').FORMAT_XZ,memlimit=64*1024*1024);raw=decomp.decompress(data[p33.HEADER.size:],max_length=rawlen+1)
    mode_len=math.ceil(n*3/8);modes=p33.unpack_modes(raw[:mode_len],n);params=raw[mode_len:];cursor=0;dc=detail=0
    for mode in modes:
        if mode==1:dc+=3;cursor+=3
        elif mode==4:detail+=48;cursor+=48
    if cursor!=len(params):raise ValueError('control byte accounting mismatch')
    return {'header_bytes':p33.HEADER.size,'mode_bytes':mode_len,'dc_bytes':dc,'detail_grid_bytes':detail,'raw_parameter_bytes':len(params),'raw_payload_bytes':len(raw),'xz_payload_bytes':xz,'xz_vs_raw_delta_bytes':xz-len(raw),'cell_mode_counts':{str(int(m)):int(np.count_nonzero(modes==m)) for m in (0,1,4)}}

def main():
    selection=json.loads((HERE/'dataset_selection.json').read_text());bases=json.loads((OUT/'base_manifest.json').read_text())
    base_rows={r['image']:r for r in bases['images']};base_dec={r['image']:r for r in json.loads((OUT/'base_decode_verification.json').read_text())['images']};all_images=[]
    for chosen in selection['images']:
        name=chosen['filename'];srcpath=ROOT/'hpt_data'/name
        source=np.asarray(ImageOps.exif_transpose(Image.open(srcpath)).convert('RGB'),np.uint8)
        baseinfo=base_rows[name];base=np.load(OUT/'bases'/f'{name}.base.npy',allow_pickle=False)
        if source.shape!=base.shape or sha(srcpath.read_bytes())!=chosen['source_sha256']:raise ValueError('frozen input/base mismatch')
        edge,smooth=masks(source);rows=[];streamdir=OUT/'streams'/name;streamdir.mkdir(parents=True,exist_ok=True)
        families=[]
        for config in ('A','B'):
            started=time.perf_counter();result=p33.encode_family(source,base,config,LAMBDAS,0);wall=time.perf_counter()-started
            families.append(('control',config,0,None,result,wall,len(result)))
        for band in BANDS:
            for q in QSTEPS:
                started=time.perf_counter();result=p34.encode_family(source,base,SPARSE_LAMBDAS,q,band);wall=time.perf_counter()-started
                families.append(('sparse',band,q,None,result,wall,len(result)))
        decode_manifest=[]
        for kind,label,q,_,family,family_wall,noutputs in families:
            for result in family:
                lam=float(result['lambda']);config=label if kind=='control' else 'S'
                if kind=='control':
                    ref=p31.encode(source,base,config,lam)
                    expected=(len(ref['data']),sha(ref['modes'].tobytes()),sha(ref['params']),sha(ref['reconstruction'].tobytes()))
                    actual=(len(result['data']),sha(result['modes'].tobytes()),sha(result['params']),sha(result['reconstruction'].tobytes()))
                    if actual!=expected:raise AssertionError(f'Pass 31 control mismatch {name} {config} {lam}')
                    accounting=p33_account(result['data']);band=None
                else:
                    accounting=p34.accounting(result['data']);band=label
                    if result['coefficients']>8*result['active_blocks']:raise AssertionError('coefficient cap violated')
                stream_name=f'{config}_{band or "A"}_q{q}_l{lam:g}.'+('p34s' if kind=='sparse' else 'p33s')
                path=streamdir/stream_name;path.write_bytes(result['data'])
                decprof={};p34.decode(result['data'],base,decprof) if kind=='sparse' else p33.decode(result['data'],base,decprof)
                times=[]
                for _ in range(REPEATS):
                    start=time.perf_counter()
                    decoded,mode,params=(p34.decode(result['data'],base) if kind=='sparse' else p33.decode(result['data'],base))
                    times.append(time.perf_counter()-start)
                row={'key':f'{config}|{band or "-"}|q{q}|lambda{lam:g}','image':name,'config':config,'band':band,'qstep':q,'lambda':lam,
                    'base_bytes':baseinfo['base_bytes'],'enhancement_bytes':len(result['data']),'total_bytes':baseinfo['base_bytes']+len(result['data']),
                    'total_bpp':(baseinfo['base_bytes']+len(result['data']))*8/(source.shape[0]*source.shape[1]),
                    'metrics':metrics(source,result['reconstruction'],edge,smooth),'mode_counts':result.get('mode_counts'),
                    'active_cells':int(np.count_nonzero(result['modes'])),'stream_path':str(path.relative_to(HERE)),'stream_sha256':sha(result['data']),
                    'pixels_sha256':sha(result['reconstruction'].tobytes()),'modes_sha256':sha(result['modes'].tobytes()),'params_sha256':sha(result['params']),
                    'family_wall_seconds':family_wall,'family_analysis_seconds':result['family_analysis_seconds'],
                    'analysis_amortized_seconds':result['family_analysis_seconds']/noutputs,'serialization_amortized_seconds':(family_wall-result['family_analysis_seconds'])/noutputs,
                    'decode_seconds':{'samples':times,'median':statistics.median(times),'mean':statistics.mean(times)},
                    'decode_profile':decprof,**accounting,'lambda':lam}
                rows.append(row)
                decode_manifest.append({'key':row['key'],'stream':str(path),'type':'p34' if kind=='sparse' else 'p33','base_npy':str(OUT/'bases'/f'{name}.base.npy'),'expected_pixels_sha256':row['pixels_sha256'],'expected_modes_sha256':row['modes_sha256'],'expected_params_sha256':row['params_sha256']})
            print('encoded family',name,label,q,len(family),f'{family_wall:.1f}s',flush=True)
        mf=streamdir/'decode_manifest.json';mf.write_text(json.dumps({'files':decode_manifest},indent=2)+'\n')
        proc=subprocess.run([sys.executable,'-B',str(HERE/'worker_decode.py'),str(mf)],cwd=ROOT,check=True,capture_output=True,text=True)
        fresh=json.loads(proc.stdout)['rows'];freshby={r['key']:r for r in fresh}
        for row in rows:
            check=freshby[row['key']]
            if check['pixels_sha256']!=row['pixels_sha256'] or check['modes_sha256']!=row['modes_sha256'] or check['params_sha256']!=row['params_sha256']:
                raise AssertionError('fresh process hash mismatch '+row['key'])
            row['fresh_process_verified']=True;row['fresh_process_peak_working_set_bytes']=check.get('peak_working_set_bytes')
            row['fresh_decode_profile']=check['profile']
            rec=np.load(OUT/'reconstructions'/name/f'{check["key"].replace("|","_")}.npy',allow_pickle=False)
            if sha(rec.tobytes())!=row['pixels_sha256']:raise AssertionError('fresh reconstruction changed on disk')
        all_images.append({'image':name,'width':source.shape[1],'height':source.shape[0],'base_bytes':baseinfo['base_bytes'],'base_stream_sha256':baseinfo['base_stream_sha256'],'base_pixels_sha256':base_dec[name]['pixels_sha256'],'rows':rows,'fresh_process_verified_count':len(fresh)})
        print('fresh-process decoded',name,len(fresh),'candidates',flush=True)
    summary={'provenance':{'starting_head':selection['selection_head'],'base_manifest_sha256':sha((OUT/'base_manifest.json').read_bytes()),'selection_manifest_sha256':sha((HERE/'dataset_selection.json').read_bytes()),'base_quality_id':'pbc31_release_rng_baseline','p33_reference_sha256':sha((ROOT/'research_pass33'/'pbc33_codec.py').read_bytes()),'p34_variant_sha256':sha((HERE/'pbc34_codec.py').read_bytes()),'frequency_bands':BANDS,'frequency_definitions':{'F':'all u,v 0..7','L':'u+v<4','H':'u+v>=4'},'quant_steps':QSTEPS,'sparse_lambdas':SPARSE_LAMBDAS,'control_lambdas':LAMBDAS,'candidate_count':sum(len(x['rows']) for x in all_images),'fresh_process_verified_count':sum(x['fresh_process_verified_count'] for x in all_images),'thread_limit':1},'images':all_images}
    (OUT/'codec_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PBC candidates',summary['provenance']['candidate_count'],'fresh',summary['provenance']['fresh_process_verified_count'],flush=True)
if __name__=='__main__':main()
