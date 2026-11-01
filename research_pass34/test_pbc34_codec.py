from __future__ import annotations
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'research_pass33'))
sys.path.insert(1,str(HERE.parent))
import pbc33_codec as p33
import pbc34_codec as codec

class FrequencyBandTests(unittest.TestCase):
    def fixture(self,h=19,w=21):
        yy,xx=np.indices((h,w));base=np.full((h,w,3),128,np.uint8)
        src=np.empty_like(base)
        src[:,:,0]=np.clip(128+((xx*19+yy*7)%121)-60,0,255)
        src[:,:,1]=np.where((xx//3+yy//2)%2,230,20)
        src[:,:,2]=np.clip(128+((xx*3-yy*11)%81)-40,0,255)
        return src,base

    def test_eligibility_definition(self):
        for u in range(8):
            for v in range(8):
                self.assertEqual(codec.eligible(u,v,'L'),u+v<4)
                self.assertEqual(codec.eligible(u,v,'H'),u+v>=4)
                self.assertTrue(codec.eligible(u,v,'F'))

    def test_full_spectrum_matches_pass33(self):
        src,base=self.fixture(24,32)
        old=p33.encode_family(src,base,'S',(400.0,),32)[0]
        new=codec.encode_family(src,base,(400.0,),32,'F')[0]
        decoded,modes,params=codec.decode(new['data'],base)
        self.assertTrue(np.array_equal(modes,old['modes']))
        self.assertEqual(params,old['params'])
        self.assertTrue(np.array_equal(decoded,old['reconstruction']))
        self.assertEqual(new['data'][codec.HEADER.size:],old['data'][p33.HEADER.size:])

    def test_restricted_selection_eligibility_and_empty_blocks(self):
        src,base=self.fixture()
        for band in ('L','H'):
            rows=codec.encode_family(src,base,(400.0,1600.0),32,band)
            for row in rows:
                decoded,modes,params=codec.decode(row['data'],base)
                self.assertTrue(np.array_equal(decoded,row['reconstruction']))
                parsed=codec.accounting(row['data'])
                self.assertEqual(parsed['band'],band)
                meta,raw=codec.unpack_raw(row['data']);n=meta['n'];ml=(n*3+7)//8
                modes2=p33.unpack_modes(raw[:ml],n);pr=raw[ml:];cursor=0
                for mode in modes2:
                    if mode==1: cursor+=3
                    elif mode==2:
                        mask=pr[cursor];cursor+=1
                        for bi in range(4):
                            if mask&(1<<bi):
                                count=pr[cursor];cursor+=1
                                for _ in range(count):
                                    pos=pr[cursor];cursor+=1;_,cursor=p33._read_varint(pr,cursor)
                                    rem=pos%64;u,v=divmod(rem,8)
                                    self.assertTrue(codec.eligible(u,v,band))
                self.assertEqual(cursor,len(pr))

    def test_decoder_rejects_forbidden_position(self):
        base=np.zeros((16,16,3),np.uint8);modes=np.array([2],np.uint8)
        params=bytes((1,1,63,2)) # block mask/count; position (7,7,R); zigzag(1)
        data=codec.wrap_raw(16,16,16,'L',modes,params)
        with self.assertRaisesRegex(ValueError,'frequency band'):
            codec.decode(data,base)
        good=codec.wrap_raw(16,16,16,'H',modes,params)
        codec.decode(good,base)

    def test_counterfactual_streams_recompute_activity_and_decode(self):
        src,base=self.fixture(33,41)
        full=codec.encode_family(src,base,(400.0,),32,'F')[0]
        for removed in ('L','H'):
            data=codec.counterfactual_remove(full['data'],removed)
            decoded,modes,params=codec.decode(data,base)
            self.assertEqual(codec.accounting(data)['band'],'H' if removed=='L' else 'L')
            self.assertEqual(decoded.shape,src.shape)

    def test_partial_boundary_signed_and_deterministic(self):
        src,base=self.fixture(17,25)
        src[:8,:8]=np.clip(base[:8,:8].astype(np.int16)-32,0,255).astype(np.uint8)
        a=codec.encode_family(src,base,(400.0,1600.0),16,'L')
        b=codec.encode_family(src,base,(400.0,1600.0),16,'L')
        for x,y in zip(a,b):
            self.assertEqual(x['data'],y['data'])
            decoded,_,_=codec.decode(x['data'],base)
            self.assertTrue(np.array_equal(decoded,x['reconstruction']))

    def test_resource_and_truncation_rejected(self):
        src,base=self.fixture()
        data=codec.encode_family(src,base,(400.0,),64,'H')[0]['data']
        for malformed in (data[:-1],data+b'x'):
            with self.assertRaises(ValueError): codec.decode(malformed,base)

    def test_fresh_process_file_only_decode(self):
        src,base=self.fixture(19,21)
        row=codec.encode_family(src,base,(400.0,),16,'F')[0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);np.save(root/'base.npy',base,allow_pickle=False);(root/'stream.p34s').write_bytes(row['data'])
            expected=hashlib.sha256(row['reconstruction'].tobytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'files':[{'key':'fixture','type':'p34','base_npy':str(root/'base.npy'),'stream':str(root/'stream.p34s'),'expected_pixels_sha256':expected,'expected_modes_sha256':hashlib.sha256(row['modes'].tobytes()).hexdigest(),'expected_params_sha256':hashlib.sha256(row['params']).hexdigest()}]}))
            p=subprocess.run([sys.executable,'-B',str(HERE/'worker_decode.py'),str(root/'manifest.json')],cwd=HERE.parent,capture_output=True,text=True,check=False)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)['rows'][0]['pixels_sha256'],expected)

if __name__=='__main__': unittest.main()
