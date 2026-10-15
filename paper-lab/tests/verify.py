"""Cross-language, invariants and deterministic sampling; no quality claims."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from classical import inpaint
from onnx_inpaint import pack_inputs

root=Path(__file__).resolve().parents[2]
cases=[]
for shape in ('center','edge','crack','disconnected','empty'):
    h,w=13,17
    yy,xx=np.indices((h,w))
    image=np.stack([(xx*13+yy*3)%256,(yy*17)%256,((xx//3+yy//3)%2)*180+30],axis=-1).astype(np.uint8)
    mask=np.zeros((h,w),np.uint8)
    if shape=='center':mask[5:8,7:10]=1
    if shape=='edge':mask[:3,:2]=1
    if shape=='crack':mask[2:11,8]=1
    if shape=='disconnected':mask[1,1]=1;mask[10:12,13:15]=1
    for method in ('bertalmio','telea','efros-leung','criminisi','patchmatch','feature'):
        options={'patch':3} if method in ('efros-leung','criminisi','patchmatch','feature') else {}
        if method=='bertalmio':options={'iterations':20}
        if method=='patchmatch':options['iterations']=3
        cases.append(dict(method=method,rgb=image.ravel().tolist(),mask=mask.ravel().tolist(),w=w,h=h,options=options))
result=subprocess.run(['node',str(root/'paper-lab/tests/run-js.mjs')],input=json.dumps(cases),capture_output=True,text=True,check=True)
js=json.loads(result.stdout)
for case,actual in zip(cases,js):
    assert 'error' not in actual,(case['method'],actual)
    image=np.array(case['rgb'],np.uint8).reshape(case['h'],case['w'],3)
    mask=np.array(case['mask'],np.uint8).reshape(case['h'],case['w'])
    py=inpaint(case['method'],image,mask,**case['options'])
    output=np.array(actual['image'],np.uint8).reshape(image.shape)
    delta=np.abs(output.astype(int)-py.astype(int)).max()
    assert delta<=1,(case['method'],'JS/Python max error',delta)
    assert np.array_equal(py[mask==0],image[mask==0]),case['method']
    poisoned=image.copy();poisoned[mask!=0]=255-image[mask!=0]
    rerun=inpaint(case['method'],poisoned,mask,**case['options'])
    assert np.array_equal(py,rerun),(case['method'],'masked input leaked')
    assert np.array_equal(image.ravel(),case['rgb']),'input mutated'
    assert np.array_equal(inpaint(case['method'],image,mask,**case['options']),py),'nondeterministic'
# Tensor polarity/layout parity with JS smoke assertion.
rgb=np.array([[[11,22,33],[44,55,66]]],np.uint8);m=np.array([[0,1]],np.uint8)
p=pack_inputs('migan',rgb,m)
assert p['image'].ravel().tolist()==[11,0,22,0,33,0]
assert p['mask'].ravel().tolist()==[255,0]
p=pack_inputs('lama',rgb,m)
assert p['mask'].ravel().tolist()==[0.,1.]
assert p['image'].dtype==np.float32
for method in ('bertalmio','telea','efros-leung','criminisi','patchmatch','feature'):
    try:inpaint(method,np.zeros((7,7,3),np.uint8),np.ones((7,7),np.uint8),patch=3)
    except ValueError:pass
    else:raise AssertionError('full mask accepted')
print(f'PASS: {len(cases)} JS/Python cases; known-pixel preservation, masked-input independence, deterministic seeds, invalid masks and ONNX tensor contracts')
