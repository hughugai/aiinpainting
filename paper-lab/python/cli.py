import argparse
import numpy as np
from PIL import Image
from classical import inpaint
from onnx_inpaint import inpaint_onnx

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Paper Lab: white mask = hole')
    p.add_argument('method',choices=['bertalmio','telea','efros-leung','criminisi','patchmatch','feature','lama','migan'])
    p.add_argument('image');p.add_argument('mask');p.add_argument('output')
    p.add_argument('--model',help='ONNX model path for lama/migan')
    p.add_argument('--patch',type=int,default=5)
    p.add_argument('--seed',type=int,default=1)
    args=p.parse_args()
    image=np.asarray(Image.open(args.image).convert('RGB'))
    mask=np.asarray(Image.open(args.mask).convert('L'))>127
    if args.method in ('lama','migan'):
        if not args.model:p.error('--model required')
        result=inpaint_onnx(args.method,args.model,image,mask)
    else:
        opts={'patch':args.patch} if args.method in ('efros-leung','criminisi','patchmatch','feature') else {}
        if args.method in ('efros-leung','patchmatch'):opts['seed']=args.seed
        result=inpaint(args.method,image,mask,**opts)
    Image.fromarray(result).save(args.output)
