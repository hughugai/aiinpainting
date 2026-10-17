"""Optional actual model CPU/WASM parity. Requires a downloaded graph path."""
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from onnx_inpaint import inpaint_onnx
method,path=sys.argv[1:]
root=Path(__file__).resolve().parents[2]
subprocess.run(['node','paper-lab/tests/onnx-real.mjs',method,path],cwd=root,check=True)
w,h=64,48
image=np.zeros((h,w,3),np.uint8)
mask=np.zeros((h,w),np.uint8)
for y in range(h):
    for x in range(w):
        image[y,x]=[x*3,y*4,120]
        if 27<x<36 and 20<y<28:mask[y,x]=1
cpu=inpaint_onnx(method,path,image,mask)
wasm=np.fromfile('/private/tmp/inpainting-'+method+'-wasm.rgb',np.uint8).reshape(h,w,3)
delta=np.abs(cpu.astype(int)-wasm.astype(int))
assert np.array_equal(cpu[mask==0],image[mask==0])
assert int(delta.max())<=3,(method,int(delta.max()))
print(method,'CPU/WASM parity PASS, maximum byte difference:',int(delta.max()))
