import unittest
import numpy as np
import phase_a_support as e

class PhaseATests(unittest.TestCase):
    def test_signed_rounding_and_pattern_interpolation(self):
        p=np.zeros((4,4,3),np.int16); p[0,0]=[-8,8,-16]; p[3,3]=[16,-8,8]
        a=e.bilinear(p); b=e.bilinear(p)
        self.assertEqual(a.shape,(16,16,3)); self.assertTrue(np.array_equal(a,b))
        self.assertTrue(np.array_equal(a[0,0],p[0,0])); self.assertTrue(np.array_equal(a[-1,-1],p[-1,-1]))
        self.assertTrue(np.array_equal(e.bilinear(np.full((4,4,3),-16,np.int16)),np.full((16,16,3),-16,np.int16)))
        self.assertEqual(e.rnd(np.array([-1.5,-.5,0,.5,1.5])).tolist(),[-2,-1,0,1,2])

    def test_quantized_fit_keeps_signed_coefficients(self):
        low=np.zeros((12,4,4,3),np.float32); pattern=np.zeros((4,4,3),np.int16)
        pattern[1:3,1:3]=[16,-8,8]
        up=e.bilinear(pattern).astype(np.float32).reshape(4,4,4,4,3).mean((1,3))
        low[:4]=up; low[4:8]=-up; low[8:]=0
        dc=np.zeros((12,3),np.int16); assign=np.arange(12,dtype=np.uint8)%4
        pats,coef=e.fit(low,dc,assign,4,8)
        self.assertEqual(pats.shape,(4,4,4,3)); self.assertTrue(np.all(pats%8==0))
        self.assertTrue(set(np.unique(coef)).issubset(set(range(7))))
        self.assertTrue({1,3,5}.issubset(set(coef.tolist())))

    def test_partial_cells_and_disconnected_assignment(self):
        base=np.full((21,35,3),100,np.uint8); residual=np.zeros_like(base,dtype=np.int16)
        rc,valid,rows,cols=e.cells(residual)
        self.assertEqual((rows,cols),(2,3)); self.assertEqual(valid.sum(),21*35)
        pats=np.zeros((2,4,4,3),np.int16); pats[1]=16
        dc=np.zeros((len(rc),3),np.int16); groups=np.array([1,0,1,0,1,0],np.uint8)
        coef=np.array([6,3,0,3,6,3],np.uint8)
        rec=e.reconstruct(base,dc,pats,coef,groups)
        self.assertTrue(np.all(rec[:16,:16]>100))
        self.assertTrue(np.all(rec[:16,32:]<100))
        self.assertTrue(np.all(rec[16:,:16]==100))
        self.assertTrue(np.all(rec[16:,16:32]>100))
        self.assertTrue(np.array_equal(rec[:16,16:32],base[:16,16:32]))
        self.assertEqual(rec.shape,base.shape)

if __name__=='__main__': unittest.main()
