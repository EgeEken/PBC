import tracemalloc
from common import *


def main():
    setup();rows=[]
    for name in CASES:
        data,record=saved(name)
        (w,h,step,n),base,payload=reference.split(data)
        q=reference.read_coefficients(payload,w,h,n)
        tracemalloc.start()
        scaled=q.astype(np.float32)*step
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        rows.append(dict(image=name,operation='original_dequant_expression',output_bytes=scaled.nbytes,traced_current=current,traced_peak=peak))
        del scaled
        pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
        color=reference.h17.correction(q,step,h,w)
        tracemalloc.start()
        expected=np.clip(np.rint(pixels+color),0,255).astype(np.uint8)
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        rows.append(dict(image=name,operation='original_final_expression',output_bytes=expected.nbytes,traced_current=current,traced_peak=peak))
        base_uint8=pixels.astype(np.uint8)
        tracemalloc.start()
        np.add(color,base_uint8,out=color);np.rint(color,out=color);np.clip(color,0,255,out=color);output=color.astype(np.uint8)
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        rows.append(dict(image=name,operation='reuse_final_expression',output_bytes=output.nbytes,traced_current=current,traced_peak=peak))
        assert np.array_equal(output,expected) and sha(output.tobytes())==record['pixels_sha256']
    write('allocations.json',dict(provenance=meta(),rows=rows,note='One expression-level tracemalloc probe per case, after inputs prepared. Includes buffers registered by NumPy with Python tracing, not all native process memory. Does not replace isolated RSS peaks or timing. Original dequant expression measured intact to detect actual temporary reuse. Final reuse probe starts with existing correction and uint8 base and counts output/new temporaries.'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
