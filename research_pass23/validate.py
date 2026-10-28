import subprocess
import sys
from common import *
from research_pass23.encoder import DecoderAwarePBC,PARAMETERS
from research_pass23.working_target import adjust,reconstruct
from research_pass2.stream_tools import parse


def main():
    setup();checked=[];fixtures=[]
    screen=json.loads((HERE/'screen.json').read_text());assert not screen['gate']['passed']
    for row in screen['rows']:
        data=(HERE/row['file']).read_bytes();assert sha(data)==row['stream_sha256'] and len(data)==row['bytes']
        output=np.asarray(PBC3.decompress(data).image.convert('RGB'))
        assert np.array_equal(output,np.asarray(PBC3.decompress(data).image.convert('RGB')))
        assert sha(output.tobytes())==row['pixels_sha256'] and output.shape==(row['source_size'][1],row['source_size'][0],3)
        error=np.asarray(load(row['image']),dtype=np.float32)-output.astype(np.float32)
        mse=float(np.mean(error**2,dtype=np.float64));assert mse==row['rgb_mse']==row['internal_mse']
        header,patches,ledger=parse(data);assert len(patches)==row['total_patches'] and not header[12]
        assert 8*len(data)/(row['source_size'][0]*row['source_size'][1])==row['bpp']
        checked.append(dict(image=row['image'],preset=row['preset'],variant=row['variant'],stream_sha256=sha(data),pixels_sha256=sha(output.tobytes()),rgb_mse=mse,v0=True,repeat_decode_exact=True))
    rng=np.random.default_rng(2003)
    for size in ((1,1),(17,13),(35,37)):
        w,h=size
        for color in ('RGB','YCbCr'):
            image=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
            for cap in (99,250000):
                config=PBC3Config.quality(auto_downsample_max_pixels=cap,color_space=color)
                original=PBC3.prepare(image,config);prepared=DecoderAwarePBC.prepare(image,config)
                repeated=DecoderAwarePBC.prepare(image,config)
                assert prepared['target'].dtype==np.int32 and prepared['arr'].dtype==np.uint8 and np.array_equal(prepared['arr'],prepared['target'])
                assert np.array_equal(prepared['target'],repeated['target'])
                assert prepared['orig_compare'].tobytes()==image.tobytes() and prepared['orig_compare'].size==image.size
                assert prepared['target'].shape==original['target'].shape and np.max(np.abs(prepared['target']-original['target']))<=8
                if not original['downsampled']:assert np.array_equal(prepared['target'],original['target'])
                assert np.array_equal(prepared['target'],adjust(original,*PARAMETERS))
                expected=PBC3._canvas_to_image(prepared['target'],color,False)
                if original['downsampled']:expected=expected.resize(image.size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
                assert np.array_equal(reconstruct(prepared['target'],prepared),np.asarray(expected))
                fixtures.append(dict(size=list(size),color=color,cap=cap,working_size=[prepared['w'],prepared['h']],target_sha256=sha(prepared['target'].tobytes())))
    rejected=0
    for image,config in ((Image.new('RGBA',(17,13)),PBC3Config.quality()),(Image.new('RGB',(35,37)),PBC3Config.quality(auto_downsample_max_pixels=99,warmup_ratio=.5))):
        try:DecoderAwarePBC.prepare(image,config)
        except ValueError:rejected+=1
        else:raise AssertionError('unsupported research configuration accepted')
    code="import sys,hashlib;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root));sys.path.insert(0,str(root/'research_pass1/release') if sys.argv[3]=='release' else str(root));from PBC3 import PBC3;from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('model loaded')));print(hashlib.sha256(PBC3.decompress(Path(sys.argv[2]).read_bytes()).image.convert('RGB').tobytes()).hexdigest())"
    for name in ('googlewiki.png','cat.jpg'):
        row=next(r for r in screen['rows'] if r['image']==name and r['preset']=='quality' and r['variant']=='candidate')
        for source in ('practical','release'):
            result=subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(HERE/row['file']),source],cwd=ROOT,text=True).strip()
            assert result==row['pixels_sha256']
    write('validation.json',dict(provenance=meta(),files=checked,prepare_fixtures=fixtures,unsupported_alpha_warmup_rejections=rejected,fresh_process_model_forbidden=4,independent_decoders=['unchanged practical PBC3 v0','immutable PBC3.0 release runtime research_pass1/release'],no_extra_image_encodes=True,scope='Sixteen saved screen streams; same original RGB internal/external MSE, ordinary stream magic, unchanged model/decoder syntax.12 deterministic prepare/reconstruction fixtures including RGB/YCbCr, identity/downsized and odd/1x1 dimensions. Decoder-equivalent helper explicitly matches production clip/cast/color/resize operations. No production global monkeypatch.'))
    print('validated',len(checked),'v0 files',len(fixtures),'prepare fixtures',rejected,'unsupported configuration rejections and four independent model-free subprocesses',flush=True)


if __name__=='__main__':main()
