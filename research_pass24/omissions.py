from common import *


def main():
    setup();ARTIFACTS.mkdir(exist_ok=True);files=[];rows=[];all_patches=[]
    for name in IMAGES:
        source=np.asarray(load(name),dtype=np.uint8)
        for preset in PRESETS:
            data,record=saved(name,preset)
            header,patches,accounting=parse(data)
            assert serialize(header,patches)==data
            output=np.asarray(PBC3.decompress(data).image.convert('RGB'))
            assert sha(output.tobytes())==record['pixels_sha256'] and rgb_mse(source,output)==record['rgb_mse']
            descriptions_rows=descriptions(header,patches,accounting)
            selected=select(descriptions_rows,header[6])
            baseline=dict(image=name,preset=preset,bytes=len(data),rgb_mse=record['rgb_mse'],stream_sha256=sha(data),pixels_sha256=record['pixels_sha256'],input_sha256=sha((ROOT/'hpt_data'/name).read_bytes()),config=record['config'],total_patches=len(patches),initial_patches=header[6],iterative_patches=len(patches)-header[6],selected=selected,raw_bytes=accounting['raw_bytes'],original_file=record['file'],exact_roundtrip=True)
            files.append(baseline)
            for row in descriptions_rows:all_patches.append(dict(image=name,preset=preset,**row,sampled=row['patch'] in selected))
            for index in selected:
                assert index>=header[6]
                candidate=patches[:index]+patches[index+1:]
                changed,pixels,result=evaluate(header,candidate,source)
                row=dict(image=name,preset=preset,**descriptions_rows[index],**result,baseline_bytes=len(data),baseline_rgb_mse=record['rgb_mse'],byte_change=result['bytes']-len(data),mse_change=result['rgb_mse']-record['rgb_mse'],rate_change_percent=100*(result['bytes']/len(data)-1),mse_change_percent=100*(result['rgb_mse']/record['rgb_mse']-1),stream_sha256=sha(changed),pixels_sha256=sha(pixels.tobytes()),remaining_patches=len(candidate),bpp=8*len(changed)/(source.shape[0]*source.shape[1]))
                if row['byte_change']<0 and row['mse_change']<0:category='strict_rd_improvement'
                elif row['byte_change']<0 and row['mse_change']==0:category='rate_saving_no_mse_loss'
                elif row['byte_change']<0 and 0<row['mse_change_percent']<=1:category='rate_saving_tradeoff'
                elif abs(row['rate_change_percent'])<.1 and abs(row['mse_change_percent'])<.01:category='no_useful_change'
                else:category='counterproductive_outside_low_loss_screen'
                row['category']=category
                path=ARTIFACTS/(name+'.'+preset+f'.omit{index}.pbc');path.write_bytes(changed);row['file']=str(path.relative_to(HERE))
                rows.append(row)
            subset=[r for r in rows if r['image']==name and r['preset']==preset]
            print(name,preset,'samples',len(subset),'strict',sum(r['category']=='strict_rd_improvement' for r in subset),'low_loss_5pct',sum(r['rate_change_percent']<=-5 and r['mse_change_percent']<=1 for r in subset),flush=True)
    strict=[r for r in rows if r['category']=='strict_rd_improvement' and r['rate_change_percent']<=-.5]
    savings=[r for r in rows if r['rate_change_percent']<=-5 and r['mse_change_percent']<=1]
    gate=dict(strict_halfpercent_count=len(strict),strict_halfpercent_images=sorted(set(r['image'] for r in strict)),fivepercent_low_loss_count=len(savings),fivepercent_low_loss_images=sorted(set(r['image'] for r in savings)))
    gate['passed']=(len(strict)>=3 and len(gate['strict_halfpercent_images'])>=2) or len(gate['fivepercent_low_loss_images'])>=2
    write('omissions.json',dict(provenance=meta(),files=files,patch_ledger=all_patches,rows=rows,gate=gate,timing='One warmed exact omitted-file serialization + production entropy + independent ordinary RGB decode + source scoring per sampled iterative patch. Parse/overlap/selection, disk/hashes excluded from evaluation_seconds and recorded as offline analysis scope. Complete files reused; no image encode. Same signed canvas and final decoder color/resize semantics.'))
    print('gate',gate,flush=True)


if __name__=='__main__':main()
