"""Cross-language parity and invariants; not a reproduction of paper PSNR."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inpaint import inpaint
ROOT=Path(__file__).resolve().parent

def js(image,mask,options):
    payload={'rgb':image.reshape(-1).tolist(),'mask':mask.reshape(-1).astype(int).tolist(),
             'width':image.shape[1],'height':image.shape[0],'options':options}
    return json.loads(subprocess.check_output(['node',str(ROOT/'run-json.mjs')],input=json.dumps(payload).encode(),stderr=subprocess.PIPE))

def compare(image,mask,method='feature',update='raw',patch=5,lam=.35):
    out,t=inpaint(image,mask,patch=patch,method=method,confidence_update=update,lam=lam)
    b=js(image,mask,{'patch':patch,'method':method,'confidenceUpdate':update,'lambda':lam})
    assert np.array_equal(out,np.array(b['image'],dtype=np.uint8).reshape(image.shape))
    assert np.array_equal(out[~mask],image[~mask]),'Known pixels changed'
    assert sum(s['filled'] for s in t)==mask.sum(),'Not all masked pixels filled'
    assert len(t)==len(b['trace'])
    for a,z in zip(t,b['trace']):
        for key in ('center','source','filled'): assert a[key]==z[key],(key,a,z)
        for key in ('confidence','data','feature','refined','priority','ssd'):
            assert abs(a[key]-z[key])<1e-9,(key,a,z)
    poison=image.copy();poison[mask]=np.random.default_rng(99).integers(0,256,(int(mask.sum()),3),dtype=np.uint8)
    clean,tt=inpaint(poison,mask,patch=patch,method=method,confidence_update=update,lam=lam)
    assert np.array_equal(out,clean),'Masked RGB leaked into computation'
    return out,t

def main():
    cases=0
    rng=np.random.default_rng(21)
    for kind in ('texture','flat','edge','border','crack','empty'):
        image=rng.integers(0,256,(15,19,3),dtype=np.uint8)
        if kind=='flat': image[:]=[70,120,180]
        if kind=='edge': image[:,:9]=[20,30,40];image[:,9:]=[210,220,230]
        mask=np.zeros((15,19),bool)
        if kind=='border': mask[:4,:3]=True
        elif kind=='crack':mask[7,3:16]=True
        elif kind!='empty':mask[5:10,7:12]=True
        for method in ('feature','shifted','criminisi'):
            compare(image,mask,method);cases+=1
    compare(image,np.zeros((15,19),bool),update='refined');cases+=1
    image=rng.integers(0,256,(15,19,3),dtype=np.uint8);mask=np.zeros((15,19),bool);mask[5:10,7:12]=True
    compare(image,mask,update='refined');cases+=1
    out,t=compare(image,mask,method='shifted',lam=2);assert any(s['priority']<0 for s in t);cases+=1
    # Same strict errors on full-hole, no source, invalid patch, iteration limit.
    no_source=np.ones((15,19),bool);no_source[7,9]=False
    for m,p,limit in ((np.ones((15,19),bool),5,300),(no_source,5,300),(mask,4,300),(mask,5,0)):
        a=False;b=False
        try:inpaint(image,m,patch=p,max_iterations=limit)
        except (ValueError,RuntimeError):a=True
        try:js(image,m,{'patch':p,'maxIterations':limit})
        except subprocess.CalledProcessError:b=True
        assert a and b;cases+=1
    # Deterministic reference scene with repeated horizontal structures.
    im=Image.new('RGB',(64,48),(170,200,220));draw=ImageDraw.Draw(im)
    for y in range(0,48,8):draw.rectangle((0,y,63,y+2),fill=(30,65,95))
    arr=np.array(im);m=np.zeros((48,64),bool);m[17:30,26:39]=True
    Path(ROOT/'examples').mkdir(exist_ok=True)
    im.save(ROOT/'examples/original.png');Image.fromarray(m.astype(np.uint8)*255).save(ROOT/'examples/mask.png')
    damaged=arr.copy();damaged[m]=[255,65,65];Image.fromarray(damaged).save(ROOT/'examples/damaged.png')
    metrics={}
    for method in ('feature','criminisi'):
        out,t=compare(arr,m,method,patch=9);cases+=1
        Image.fromarray(out).save(ROOT/f'examples/{method}.png')
        mse=float(np.mean((out[m].astype(float)-arr[m])**2));metrics[method]={'iterations':len(t),'masked_mse':mse,'masked_psnr_db':None if mse==0 else float(10*np.log10(255**2/mse)),'exact_mask_recovery':mse==0}
    panel=Image.new('RGB',(64*4,48))
    for j,name in enumerate(('original','damaged','criminisi','feature')):panel.paste(Image.open(ROOT/f'examples/{name}.png'),(j*64,0))
    panel.resize((1024,192),Image.Resampling.NEAREST).save(ROOT/'examples/comparison.png')
    report={'passed_cases':cases,'rgb_parity':'exact','trace_numeric_tolerance':1e-9,'known_pixels':'unchanged','masked_input_values':'ignored','synthetic_metrics':metrics,'paper_benchmark':'not reproduced: original images/masks unavailable'}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
