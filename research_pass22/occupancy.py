from common import *


def main():
    setup();rows=[]
    for name in CASES:
        data,record=saved(name)
        (w,h,step,n),base,payload=reference.split(data)
        q=reference.read_coefficients(payload,w,h,n)
        counts=np.count_nonzero(q,axis=(-2,-1))
        occupied=counts>0
        spatial=np.any(occupied,axis=2)
        heights=np.minimum(8,h-np.arange(q.shape[0])*8)
        widths=np.minimum(8,w-np.arange(q.shape[1])*8)
        covered=int(np.sum(spatial*heights[:,None]*widths[None,:]))
        grid=[]
        for ys in np.array_split(np.arange(q.shape[0]),4):
            grid.append([float(spatial[np.ix_(ys,xs)].mean()) for xs in np.array_split(np.arange(q.shape[1]),4)])
        row=dict(image=name,step=step,stream_sha256=record['stream_sha256'],pixels_sha256=record['pixels_sha256'],size=[w,h],coefficient_sha256=sha(q.tobytes()),spatial_blocks=int(spatial.size),component_blocks=int(counts.size),component_count_histogram={'zero':int(np.sum(counts==0)),'one':int(np.sum(counts==1)),'two_to_four':int(np.sum((counts>=2)&(counts<=4))),'over_four':int(np.sum(counts>4))},component_nonzero_counts=np.count_nonzero(occupied,axis=(0,1)).tolist(),spatial_zero_blocks=int(np.sum(~spatial)),component_zero_fraction=float(np.mean(~occupied)),spatial_zero_fraction=float(np.mean(~spatial)),source_pixel_coverage=covered/(w*h),spatial_occupied_fraction_4x4=grid,channel_occupied_fraction_4x4=[[[float(occupied[np.ix_(ys,xs,[c])].mean()) for xs in np.array_split(np.arange(q.shape[1]),4)] for ys in np.array_split(np.arange(q.shape[0]),4)] for c in range(3)])
        rows.append(row);print(name,row['component_zero_fraction'],row['spatial_zero_fraction'],row['source_pixel_coverage'],flush=True)
    gate=sum(r['spatial_zero_fraction']>=.2 for r in rows)>=2
    write('occupancy.json',dict(provenance=meta(),rows=rows,sparse_idct_gate=gate,rule='At least20% empty spatial blocks on at least two of three cases; per-channel savings reported separately.'))


if __name__=='__main__':main()
