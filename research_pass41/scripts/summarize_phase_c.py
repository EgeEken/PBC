import json,pathlib,statistics
p=pathlib.Path('research_pass41/outputs/phase_c_results.json'); d=json.loads(p.read_text()); rows=d['rows']; phase_b=json.loads((p.parent/'phase_b_results.json').read_text())['rows']
points=[('googlewiki.png',5,'low'),('googlewiki.png',20,'high'),('cat.jpg',35,'low'),('cat.jpg',50,'high'),('coast.jpg',80,'low'),('coast.jpg',90,'high')]
out=[]
for image,q,label in points:
  config_rows={}
  for config in ('A','B'):
    if label=='low': phase='phaseB_timed' if config=='A' else 'phaseC_low_timed'
    else: phase='phaseC_high_timed'
    source_rows=phase_b if label=='low' and config=='A' else rows
    rr=[x for x in source_rows if x.get('image')==image and x.get('qcolor')==q and x.get('config')==config and x.get('phase')==phase and x.get('repeat') in (1,2,3)]
    if len(rr)!=3: raise ValueError((image,q,config,phase,len(rr)))
    a=rr[0]
    config_rows[config]={
      'bytes':a['bytes'],'bpp':a['bpp'],'rgb_mse':a['rgb_mse'],'rgb_psnr':a['rgb_psnr'],'ssimulacra2':a['ssimulacra2'],
      'encode_seconds_median':statistics.median(x['encode_seconds'] for x in rr),'encode_seconds_samples':[x['encode_seconds'] for x in rr],
      'decode_seconds_median':statistics.median(x['decode_seconds'] for x in rr),'decode_seconds_samples':[x['decode_seconds'] for x in rr],
      'avif_sha256':a['avif_sha256'],'decoded_rgb_sha256':a['decoded_rgb_sha256'],'source_sha256':a['source_sha256'],
      'pinned_avifdec_ok':all(x['decoder_ok'] for x in rr),'timed_repetitions':3}
    if any(x['bytes']!=a['bytes'] or x['rgb_mse']!=a['rgb_mse'] or x['avif_sha256']!=a['avif_sha256'] for x in rr): raise ValueError('nondeterministic output')
  A=config_rows['A']; B=config_rows['B']
  out.append({'image':image,'qcolor':q,'region':label,'A':A,'B':B,
    'delta':{'bytes':B['bytes']-A['bytes'],'bytes_percent':100*(B['bytes']/A['bytes']-1),'bpp_percent':100*(B['bpp']/A['bpp']-1),
      'rgb_mse':B['rgb_mse']-A['rgb_mse'],'rgb_mse_percent':100*(B['rgb_mse']/A['rgb_mse']-1),
      'rgb_psnr_db':B['rgb_psnr']-A['rgb_psnr'],'ssimulacra2':B['ssimulacra2']-A['ssimulacra2'],
      'encode_seconds_median_percent':100*(B['encode_seconds_median']/A['encode_seconds_median']-1),
      'decode_seconds_median_percent':100*(B['decode_seconds_median']/A['decode_seconds_median']-1),
      'file_sha_equal':A['avif_sha256']==B['avif_sha256'],'decoded_rgb_equal':A['decoded_rgb_sha256']==B['decoded_rgb_sha256']}})
for x in out:
 print(x['image'],x['qcolor'],f"bytes {x['delta']['bytes_percent']:+.3f}%",f"MSE {x['delta']['rgb_mse_percent']:+.3f}%",f"PSNR {x['delta']['rgb_psnr_db']:+.4f}dB",f"S2 {x['delta']['ssimulacra2']:+.4f}",f"enc {x['delta']['encode_seconds_median_percent']:+.2f}%",f"dec {x['delta']['decode_seconds_median_percent']:+.2f}%")
(p.parent/'phase_c_point_summary.json').write_text(json.dumps(out,indent=2)+'\n')
