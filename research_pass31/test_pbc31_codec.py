import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import pbc31_codec as c

class CodecTests(unittest.TestCase):
    def roundtrip(self,src,base,config='D',lam=20):
        encoded=c.encode(src,base,config,lam)
        rec,modes,params=c.decode(encoded['data'],base)
        self.assertTrue(np.array_equal(modes,encoded['modes']))
        self.assertEqual(params,encoded['params'])
        self.assertTrue(np.array_equal(rec,encoded['reconstruction']))
        return encoded,rec

    def test_pack_and_fixed_point_grid(self):
        for width in (1,2,3,4):
            values=np.arange(23,dtype=np.uint8)%(1<<width)
            packed=c.pack_bits(values,width)
            self.assertTrue(np.array_equal(c.unpack_bits(packed,len(values),width),values))
        with self.assertRaises(ValueError): c.unpack_bits(b'\xff',1,1)
        grid=np.full((4,4,3),-16,np.int8)
        self.assertTrue(np.array_equal(c.interpolate_grid(grid,16),np.full((16,16,3),-256,np.int16)))
        grid[0,0]=[8,-8,0]
        self.assertTrue(np.array_equal(c.interpolate_grid(grid,16),c.interpolate_grid(grid,16)))

    def test_all_configs_on_odd_signed_image(self):
        h,w=19,23
        y,x=np.mgrid[:h,:w]
        base=np.stack((40+x*3,90+y*2,140+((x//3+y//2)%2)*30),2).clip(0,255).astype(np.uint8)
        delta=np.zeros((h,w,3),np.int16)
        delta[:8,:10]=[24,-32,16]; delta[8:,:]=np.where((x[8:,:,None]+y[8:,:,None])%5<2,35,-20)
        src=np.clip(base.astype(np.int16)+delta,0,255).astype(np.uint8)
        for config in 'ABCD':
            encoded,rec=self.roundtrip(src,base,config,20)
            allowed=c.MASKS[config]
            self.assertTrue(all(allowed&(1<<int(mode)) for mode in encoded['modes']))
            self.assertEqual(rec.shape,(h,w,3))
            cols=math.ceil(w/16)
            self.assertTrue(np.all(np.isin(encoded['modes'][:cols],(0,1,4))))
            self.assertTrue(all(encoded['modes'][i] not in (3,6) for i in range(cols,len(encoded['modes']),cols)))

    def test_directional_prediction_uses_reconstructed_neighbors(self):
        base=np.full((32,32,3),100,np.uint8); src=np.full_like(base,124)
        encoded,rec=self.roundtrip(src,base,'C',20)
        modes=encoded['modes'].reshape(2,2)
        self.assertEqual(int(modes[0,0]),1)
        self.assertEqual(int(modes[0,1]),3)
        self.assertEqual(int(modes[1,0]),2)
        self.assertIn(int(modes[1,1]),(2,3))
        self.assertTrue(np.array_equal(src,rec))

    def test_content_fixtures(self):
        h,w=23,29; base=np.full((h,w,3),128,np.uint8)
        fixtures=[]
        fixtures.append(np.full_like(base,128))
        y,x=np.mgrid[:h,:w]; gradient=np.stack((40+x*4,50+y*5,60+(x+y)*3),2).clip(0,255).astype(np.uint8); fixtures.append(gradient)
        horizontal=base.copy(); horizontal[11:]=210; fixtures.append(horizontal)
        vertical=base.copy(); vertical[:,14:]=30; fixtures.append(vertical)
        text=Image.new('RGB',(w,h),(245,245,245)); draw=ImageDraw.Draw(text)
        draw.rectangle((2,3,5,18),fill=(12,12,12)); draw.rectangle((5,3,21,6),fill=(12,12,12))
        draw.rectangle((10,10,13,20),fill=(12,12,12)); draw.rectangle((10,15,23,18),fill=(12,12,12)); fixtures.append(np.asarray(text,np.uint8))
        for i,src in enumerate(fixtures):
            for config in 'ABCD':
                encoded,rec=self.roundtrip(src,base,config,20)
                self.assertEqual(rec.shape,(h,w,3),f'fixture {i}/{config}')

    def test_clipping_constant_and_detail_symbols(self):
        base=np.full((16,16,3),255,np.uint8); src=np.zeros_like(base)
        encoded,rec=self.roundtrip(src,base,'A',0)
        self.assertTrue(np.all(rec==0)); self.assertEqual(int(encoded['mode_counts'][1]),1)
        base2=np.zeros((17,18,3),np.uint8); src2=np.full_like(base2,255)
        self.roundtrip(src2,base2,'B',0)

    def test_malformed_streams_rejected(self):
        rng=np.random.default_rng(12); base=rng.integers(0,256,(19,23,3),dtype=np.uint8); src=base.copy(); src[2:7,3:9]=255-src[2:7,3:9]
        data=c.encode(src,base,'D',10)['data']
        for bad in (data[:5],data[:-1],data+b'x',b'NOPE'+data[4:]):
            with self.assertRaises(ValueError): c.decode(bad,base)
        fields=list(c.HEADER.unpack_from(data)); fields[2]=20000
        bad_header=c.HEADER.pack(*fields)+data[c.HEADER.size:]
        with self.assertRaises(ValueError): c.decode(bad_header,base)

    def test_fresh_process_file_only_decode(self):
        rng=np.random.default_rng(2); base=rng.integers(0,256,(21,19,3),dtype=np.uint8); src=np.roll(base,1,axis=1)
        encoded,rec=self.roundtrip(src,base,'D',20)
        worker=Path(__file__).with_name('worker_decode.py')
        temp_root=Path(__file__).with_name('outputs')/'test_temp'; temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            root=Path(temp); np.save(root/'base.npy',base,allow_pickle=False); (root/'s.p31r').write_bytes(encoded['data'])
            manifest={'base_npy':str(root/'base.npy'),'streams':[{'key':'one','stream':str(root/'s.p31r')}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            run=subprocess.run([sys.executable,'-B',str(worker),str(root/'manifest.json')],check=True,capture_output=True,text=True)
            out=json.loads(run.stdout)['rows'][0]
            self.assertEqual(out['pixels_sha256'],c.sha(rec.tobytes()))
            self.assertEqual(out['modes_sha256'],c.sha(encoded['modes'].tobytes()))
            self.assertEqual(out['params_sha256'],c.sha(encoded['params']))

if __name__=='__main__': unittest.main()
