"""Feature-precedence exemplar inpainting, Li et al. (2016). See README."""
import argparse
import json
import math
import numpy as np
from PIL import Image


def inpaint(rgb, mask, patch=9, lam=0.35, feature_weight=1.0,
            method='feature', gradient_scale=255.0, confidence_update='raw',
            max_iterations=None):
    image = np.asarray(rgb, dtype=np.uint8).copy()
    hole = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 3 or image.shape[2] != 3 or hole.shape != image.shape[:2]:
        raise ValueError('Expected HxWx3 RGB and matching HxW mask')
    h, w = hole.shape
    if patch < 3 or patch % 2 != 1 or patch > min(h, w):
        raise ValueError('patch must be odd, >=3, and fit the image')
    if method not in ('feature', 'shifted', 'criminisi') or confidence_update not in ('raw', 'refined'):
        raise ValueError('Invalid method or confidence update')
    if not all(math.isfinite(v) for v in (lam, feature_weight, gradient_scale)) or gradient_scale <= 0:
        raise ValueError('Parameters must be finite; gradient_scale must be positive')
    initial = hole.copy()
    confidence = (~hole).astype(np.float64)
    radius = patch // 2
    trace = []
    # Cache candidate coordinates per clipped patch shape. Source region is fixed.
    candidates = {}
    def sources(ph, pw):
        key = (ph, pw)
        if key not in candidates:
            candidates[key] = [(y, x) for y in range(h-ph+1) for x in range(w-pw+1)
                               if not initial[y:y+ph, x:x+pw].any()]
        return candidates[key]
    limit = max_iterations if max_iterations is not None else int(hole.sum()) + 1
    if limit < 0:
        raise ValueError('max_iterations must be nonnegative')
    for iteration in range(limit):
        if not hole.any():
            break
        gray = (image[:,:,0].astype(float)*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114)
        gx = np.zeros((h,w)); gy = np.zeros((h,w))
        # Only known neighbors: hidden RGB never contributes to priorities.
        for y in range(h):
            for x in range(w):
                if hole[y,x]:
                    continue
                left = x > 0 and not hole[y,x-1]
                right = x+1 < w and not hole[y,x+1]
                up = y > 0 and not hole[y-1,x]
                down = y+1 < h and not hole[y+1,x]
                if left and right: gx[y,x] = (gray[y,x+1]-gray[y,x-1])/2
                elif right: gx[y,x] = gray[y,x+1]-gray[y,x]
                elif left: gx[y,x] = gray[y,x]-gray[y,x-1]
                if up and down: gy[y,x] = (gray[y+1,x]-gray[y-1,x])/2
                elif down: gy[y,x] = gray[y+1,x]-gray[y,x]
                elif up: gy[y,x] = gray[y,x]-gray[y-1,x]
        magnitude = np.array([[math.hypot(gx[y,x],gy[y,x]) for x in range(w)] for y in range(h)])
        best = None
        for y in range(h):
            for x in range(w):
                if hole[y,x] or not any(hole[yy,xx] for yy,xx in
                    ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0<=yy<h and 0<=xx<w):
                    continue
                y0,y1=max(0,y-radius),min(h,y+radius+1)
                x0,x1=max(0,x-radius),min(w,x+radius+1)
                c=0.0
                for cy in range(y0,y1):
                    for cx in range(x0,x1): c+=float(confidence[cy,cx])
                c/=((y1-y0)*(x1-x0))
                # Normal = gradient of binary hole, central differences; cancellation
                # for a thin crack uses the first adjacent hole (deterministic order).
                nx=((int(hole[y,x+1]) if x+1<w else 0)-(int(hole[y,x-1]) if x>0 else 0))/2
                ny=((int(hole[y+1,x]) if y+1<h else 0)-(int(hole[y-1,x]) if y>0 else 0))/2
                norm=math.hypot(nx,ny)
                if norm==0:
                    for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                        if 0<=yy<h and 0<=xx<w and hole[yy,xx]:
                            nx,ny=xx-x,yy-y; norm=math.hypot(nx,ny); break
                nx/=norm; ny/=norm
                region=magnitude[y0:y1,x0:x1]
                k=int(region.argmax()); yy=y0+k//(x1-x0); xx=x0+k%(x1-x0)
                d=abs(-gy[yy,xx]*nx+gx[yy,xx]*ny)/255.0
                feature=float(magnitude[y,x])/gradient_scale
                refined=c if method=='criminisi' else c-lam
                if method=='feature': refined+=feature_weight*feature
                priority=refined*d
                if best is None or priority > best[0]:
                    best=(priority,y,x,y0,y1,x0,x1,c,d,feature,refined)
        if best is None:
            raise ValueError('No known fill front (the mask may cover the entire image)')
        priority,y,x,y0,y1,x0,x1,c,d,feature,refined=best
        ph,pw=y1-y0,x1-x0
        known=~hole[y0:y1,x0:x1]
        offsets=np.argwhere(known)
        target=image[y0:y1,x0:x1].astype(np.float64)
        min_ssd=math.inf; source=None
        for sy,sx in sources(ph,pw):
            # Ordered scalar accumulation matches JavaScript's tie behavior.
            ssd=0.0
            for dy,dx in offsets:
                for channel in range(3):
                    delta=float(target[dy,dx,channel])-int(image[sy+dy,sx+dx,channel])
                    ssd+=delta*delta
                if ssd>min_ssd: break
            if ssd<min_ssd: min_ssd=ssd; source=(sy,sx)
        if source is None:
            raise ValueError('No fully known source patch; reduce patch size')
        sy,sx=source
        fill=hole[y0:y1,x0:x1].copy()
        image[y0:y1,x0:x1][fill]=image[sy:sy+ph,sx:sx+pw][fill]
        confidence[y0:y1,x0:x1][fill]=c if confidence_update=='raw' else refined
        hole[y0:y1,x0:x1][fill]=False
        trace.append(dict(iteration=iteration,center=[x,y],source=[sx,sy],
                          confidence=c,data=d,feature=feature,refined=refined,
                          priority=priority,ssd=min_ssd,filled=int(fill.sum())))
    if hole.any():
        raise RuntimeError('Iteration limit reached before completion')
    return image,trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image'); p.add_argument('mask'); p.add_argument('output')
    p.add_argument('--patch',type=int,default=9)
    p.add_argument('--lambda',dest='lam',type=float,default=.35)
    p.add_argument('--feature-weight',type=float,default=1)
    p.add_argument('--gradient-scale',type=float,default=255)
    p.add_argument('--method',choices=['feature','shifted','criminisi'],default='feature')
    p.add_argument('--confidence-update',choices=['raw','refined'],default='raw')
    p.add_argument('--trace'); a=p.parse_args()
    result,trace=inpaint(np.array(Image.open(a.image).convert('RGB')),
                         np.array(Image.open(a.mask).convert('L'))>127,
                         patch=a.patch,lam=a.lam,feature_weight=a.feature_weight,
                         gradient_scale=a.gradient_scale,method=a.method,
                         confidence_update=a.confidence_update)
    Image.fromarray(result).save(a.output)
    if a.trace:
        with open(a.trace,'w') as f: json.dump(trace,f,indent=2)
    print(f'Completed {len(trace)} patches: {a.output}')

if __name__=='__main__': main()
