import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import phase_a_support as e

def signed_rgb(x): return np.clip(np.rint(128+x.astype(np.float32)*127/64),0,255).astype(np.uint8)

def main():
    name='cat.jpg'; item=e.load_case(name); src,base=item['src'],item['base']
    residual=src.astype(np.int16)-base.astype(np.int16); cells,valid,rows,cols=e.cells(residual); low=e.low_projection(cells); dc=e.qdc(low)
    fit=np.load(e.OUT/f'{name}.fit.npz'); pats=fit['residual_shape|8.patterns']; coef=fit['residual_shape|8.coeff']; assign=fit['residual_shape|8.assign']
    counts=np.bincount(assign[coef!=3],minlength=4)
    energies=np.sqrt((pats.astype(np.float32)**2).mean((1,2,3)))
    group=int(np.argmax(np.where(counts>=3,energies,-1))); ids=np.flatnonzero((assign==group)&(coef!=3))
    points=np.column_stack((ids//cols,ids%cols)); selected=[int(ids[0])]
    while len(selected)<3:
        chosen=np.array([divmod(i,cols) for i in selected]); dist=((points[:,None,:]-chosen[None,:,:])**2).sum(2).min(1)
        dist[np.isin(ids,selected)]=-1; selected.append(int(ids[int(np.argmax(dist))]))
    pattern=e.bilinear(pats[group]); rows_out=[]
    for i in selected:
        k=int(coef[i]); applied=e.rnd(e.COEFF_NUM[k]*pattern.astype(np.float32)/2).astype(np.int32)
        remaining=cells[i].astype(np.int32)-dc[i][None,None,:]-applied
        rows_out.append((i,k,cells[i],applied,remaining))
    scale=8; tile=16*scale; margin=16; header_h=50; label_h=24
    width=4*tile+5*margin; height=header_h+3*(tile+label_h+margin)
    canvas=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(canvas)
    d.text((margin,8),f'Cat / structural group {group}: one signed RGB action reused at disconnected cells (display contrast 4x)',fill='black')
    titles=('Input signed residual','Shared 4x4 pattern x4','Applied correction','Residual after DC + action')
    for j,title in enumerate(titles): d.text((margin+j*(tile+margin),header_h-18),title,fill='black')
    for row,(i,k,r,applied,remaining) in enumerate(rows_out):
        y=header_h+label_h+row*(tile+label_h+margin); x=margin
        d.text((margin,y-label_h+3),f'cell ({i//cols},{i%cols}), coefficient {e.COEFF_NUM[k]/2:g}',fill='black')
        imgs=(signed_rgb(r),signed_rgb(pattern),signed_rgb(applied),signed_rgb(remaining))
        for col,a in enumerate(imgs):
            canvas.paste(Image.fromarray(a).resize((tile,tile),Image.Resampling.NEAREST),(x+col*(tile+margin),y))
    out=e.VIS/'shared_action_cat.png'; out.parent.mkdir(parents=True,exist_ok=True); canvas.save(out)

if __name__=='__main__':main()
