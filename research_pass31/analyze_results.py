import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pbc31_codec as codec

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'; VIS=HERE/'visuals'
AVIF={
 'googlewiki.png':[{'quality':43,'bytes':80537,'mse':233.15},{'quality':84,'bytes':174211,'mse':195.64}],
 'mario64.png':[{'quality':68,'bytes':62909,'mse':50.92},{'quality':82,'bytes':88121,'mse':47.61}],
 'flower.jpg':[{'quality':18,'bytes':23171,'mse':21.06},{'quality':38,'bytes':44582,'mse':9.57}],
 'cat.jpg':[{'quality':18,'bytes':53869,'mse':115.91},{'quality':21,'bytes':75274,'mse':103.89}]}
NAMES=('googlewiki.png','mario64.png','flower.jpg','cat.jpg')
COLORS={'A':'#666666','B':'#2a78b8','C':'#e28e2c','D':'#2c9c69'}
LABELS={0:'Skip',1:'DC',2:'Vertical',3:'Horizontal',4:'Detail',5:'Vertical+DC',6:'Horizontal+DC'}

def load(): return json.loads((OUT/'summary.json').read_text())

def add_per_cell_gains(data):
    result={}
    for name,img in data['images'].items():
        for row in img['results']:
            path=OUT/'per_cell'/name/row['config']/f"{row['lambda']:g}.npz"
            with np.load(path) as z:
                modes=z['modes']; gains=z['skip_sse'].astype(np.int64)-z['chosen_sse'].astype(np.int64)
            per={}
            for mode,label in LABELS.items():
                use=modes==mode
                per[label]={'cells':int(use.sum()),'signed_sse_gain':int(gains[use].sum()),
                            'mean_sse_gain_per_used_cell':float(gains[use].mean()) if np.any(use) else None}
            row['sse_gain_by_selected_mode']=per
            result[f'{name}|{row["key"]}']=per
    return result

def comparisons(data):
    output={}; pairs=(('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D'))
    for name,img in data['images'].items():
        rows=img['results']; output[name]={'nondominated':[],'near_rate_pairs':[],'nearest_pairs':[]}
        for r in rows:
            dominated=any(x['total_bytes']<=r['total_bytes'] and x['metrics']['mse']<=r['metrics']['mse'] and
                          (x['total_bytes']<r['total_bytes'] or x['metrics']['mse']<r['metrics']['mse']) for x in rows)
            if not dominated: output[name]['nondominated'].append({'config':r['config'],'lambda':r['lambda'],
                'total_bytes':r['total_bytes'],'bpp':r['total_bpp'],'mse':r['metrics']['mse']})
        for a,b in pairs:
            left=[r for r in rows if r['config']==a]; right=[r for r in rows if r['config']==b]
            for x in left:
                y=min(right,key=lambda q:abs(q['total_bytes']/x['total_bytes']-1))
                gap=100*(y['total_bytes']-x['total_bytes'])/x['total_bytes']
                dmse=100*(y['metrics']['mse']-x['metrics']['mse'])/x['metrics']['mse']
                item={'left_config':a,'left_lambda':x['lambda'],'right_config':b,'right_lambda':y['lambda'],
                      'left_bytes':x['total_bytes'],'right_bytes':y['total_bytes'],'signed_rate_gap_pct':gap,
                      'left_mse':x['metrics']['mse'],'right_mse':y['metrics']['mse'],'right_mse_change_pct':dmse,
                      'within_5pct_rate':abs(gap)<=5}
                output[name]['nearest_pairs'].append(item)
                if abs(gap)<=5: output[name]['near_rate_pairs'].append(item)
    return output

def rate_plot(data):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,name in zip(axes.flat,NAMES):
        img=data['images'][name]
        for config in 'ABCD':
            rows=sorted((r for r in img['results'] if r['config']==config),key=lambda r:r['total_bpp'])
            ax.plot([r['total_bpp'] for r in rows],[r['metrics']['mse'] for r in rows],'-o',
                    color=COLORS[config],label=config,markersize=5)
        q=AVIF[name]
        total_pixels=img['width']*img['height']
        ax.scatter([r['bytes']*8/total_pixels for r in q],[r['mse'] for r in q],
                   marker='^',s=55,color='#b64e4e',label='AVIF Pass 26',zorder=5)
        ax.set_title(name); ax.set_xlabel('Total file rate (bits/pixel)'); ax.set_ylabel('RGB MSE')
        ax.grid(alpha=.25)
    axes.flat[0].legend(frameon=False,ncol=2)
    fig.suptitle('Pass 31 completed files; AVIF points are prior Pass 26 measurements')
    fig.savefig(VIS/'rate_distortion.png',dpi=150); plt.close(fig)

def mode_map(data):
    name='cat.jpg'; row=next(r for r in data['images'][name]['results'] if r['config']=='D' and r['lambda']==20)
    path=OUT/'per_cell'/name/'D'/'20.npz'
    with np.load(path) as z: codes=z['modes']
    h,w=data['images'][name]['height'],data['images'][name]['width']; cols=(w+15)//16; rows=(h+15)//16
    modes=codes.reshape(rows,cols); visual=np.zeros_like(modes)
    visual[np.isin(modes,(2,5))]=2; visual[np.isin(modes,(3,6))]=3
    cmap=matplotlib.colors.ListedColormap(['#b9b9b9','#3a75c4','#df7a27','#8c5eb7','#2c9c69'])
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    ax.imshow(visual,interpolation='nearest',cmap=cmap,vmin=-.5,vmax=4.5,aspect='auto')
    ax.set_title('Cat, configuration D, λ=20: selected action per 16×16 cell')
    ax.set_xlabel('Cell column'); ax.set_ylabel('Cell row')
    handles=[plt.Line2D([0],[0],marker='s',linestyle='',color=cmap(i),label=label,markersize=10)
             for i,label in enumerate(('Skip','DC','Vertical prediction','Horizontal prediction','Local detail'))]
    ax.legend(handles=handles,loc='upper right',framealpha=.94,ncol=1)
    fig.savefig(VIS/'mode_selection_cat_D_lambda20.png',dpi=150); plt.close(fig)

def main():
    VIS.mkdir(parents=True,exist_ok=True); data=load(); gains=add_per_cell_gains(data); cmp=comparisons(data)
    for name,img in data['images'].items():
        for r in img['results']:
            r['nondominated']=any(x['config']==r['config'] and x['lambda']==r['lambda'] for x in cmp[name]['nondominated'])
    rate_plot(data); mode_map(data)
    data['analysis']={'avif_pass26_points':AVIF,'rate_comparisons':cmp,
        'mode_colors_collapsed':{'2,5':'vertical','3,6':'horizontal'},
        'directional_mode_symbols':{'2':'vertical-no-offset','3':'horizontal-no-offset','5':'vertical-with-DC','6':'horizontal-with-DC'}}
    (OUT/'summary.analyzed.json').write_text(json.dumps(data,indent=2))
    compact={'provenance':data['provenance'],'config':data['config'],'avif_pass26_points':AVIF,
        'images':{},'interpretation':{'all_screen_candidates_fresh_process_verified':all(r['fresh_process_verified'] for i in data['images'].values() for r in i['results']),
            'phase_b_executed':True,'operating_points':data['config']['operating_points']}}
    for name,img in data['images'].items():
        compact['images'][name]={'class':img['class'],'width':img['width'],'height':img['height'],'base_bytes':img['base_bytes'],
            'base_sha256':img['base_sha256'],'source_sha256':img['source_sha256'],'base_mse':img['base_mse'],
            'process_peak_working_set_bytes':img['process_peak_working_set_bytes'],
            'results':[{k:r[k] for k in ('config','lambda','enhancement_bytes','total_bytes','total_bpp','enhancement_bpp','metrics',
                'mode_counts','estimated_bits','actual_enhancement_bits','actual_minus_estimated_bits','raw_payload_bytes',
                'mode_signal_raw_bytes','parameter_raw_bytes','header_bytes','entropy_payload_bytes','encode_seconds','decode',
                'directional_cells','directional_selected_sse_gain','directional_available_cell_sse_gain','sse_gain_by_selected_mode',
                'nondominated','fresh_process_verified')} for r in img['results']],
            'nondominated':cmp[name]['nondominated'],'near_rate_pairs':cmp[name]['near_rate_pairs'],
            'nearest_pairs_with_rate_gaps':cmp[name]['nearest_pairs']}
    out=HERE/'PBC_V4_PREDICTIVE_REFINEMENT_SUMMARY.json'
    out.write_text(json.dumps(compact,indent=2))
    print('fresh verified',compact['interpretation']['all_screen_candidates_fresh_process_verified'])
    for name,img in compact['images'].items():
        print(name,'nondominated',[(x['config'],x['lambda'],x['total_bytes'],round(x['mse'],3)) for x in img['nondominated']])

if __name__=='__main__':main()
