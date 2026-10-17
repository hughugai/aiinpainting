"""Independent Python counterparts of js/classical.js; see paper-lab/README.md."""
import heapq
import math
import numpy as np


def validate(rgb, mask):
    image = np.asarray(rgb)
    hole = np.asarray(mask) != 0
    if image.ndim != 3 or image.shape[2] != 3 or hole.shape != image.shape[:2]:
        raise ValueError('Expected HxWx3 RGB and HxW mask')
    if image.dtype != np.uint8:
        raise ValueError('RGB must be uint8')
    if hole.all():
        raise ValueError('At least one known pixel is required')
    return image.copy(), hole.copy()


def neighbors(i, w, h):
    x, y = i % w, i // w
    return [q for q in (i-1 if x else -1, i+1 if x<w-1 else -1,
                        i-w if y else -1, i+w if y<h-1 else -1) if q >= 0]


def byte(v):
    return max(0, min(255, math.floor(v + .5)))


def rng(seed):
    state = int(seed) & 0xffffffff
    def draw():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xffffffff
        return state / 4294967296
    return draw


def telea(rgb, mask, radius=3, **_):
    image, hole = validate(rgb, mask)
    if not isinstance(radius, int) or not 1 <= radius <= 20:
        raise ValueError('radius must be 1..20')
    h, w = hole.shape
    out = image.reshape(-1, 3)
    known = ~hole.ravel()
    times = np.where(known, 0., np.inf)
    queue = []
    serial = 0
    def offer(i):
        nonlocal serial
        x, y = i % w, i // w
        a = min(times[i-1] if x and known[i-1] else math.inf,
                times[i+1] if x<w-1 and known[i+1] else math.inf)
        b = min(times[i-w] if y and known[i-w] else math.inf,
                times[i+w] if y<h-1 and known[i+w] else math.inf)
        if not math.isfinite(a):
            t = b+1
        elif not math.isfinite(b):
            t = a+1
        else:
            t = min(a,b)+1 if abs(a-b)>=1 else (a+b+math.sqrt(2-(a-b)**2))/2
        if t < times[i]:
            times[i] = t
            heapq.heappush(queue, (t, serial, i))
            serial += 1
    def grad(values, i, step, low, high):
        l, r = low and known[i-step], high and known[i+step]
        return ((float(values[i+step])-float(values[i-step]))/2 if l and r
                else float(values[i+step])-float(values[i]) if r
                else float(values[i])-float(values[i-step]) if l else 0.)
    for i in range(w*h):
        if not known[i] and any(known[j] for j in neighbors(i,w,h)):
            offer(i)
    while queue:
        d, _, i = heapq.heappop(queue)
        if known[i] or d != times[i]:
            continue
        x, y = i%w, i//w
        nx, ny = grad(times,i,1,x>0,x<w-1), grad(times,i,w,y>0,y<h-1)
        sums, total = [0.,0.,0.], 0.
        for yy in range(max(0,y-radius), min(h,y+radius+1)):
            for xx in range(max(0,x-radius), min(w,x+radius+1)):
                q, dx, dy = yy*w+xx, x-xx, y-yy
                dist = math.hypot(dx,dy)
                if not known[q] or dist==0 or dist>radius:
                    continue
                weight = (abs(dx*nx+dy*ny)+1e-6)/(dist**3*(1+abs(d-times[q])))
                for c in range(3):
                    a = out[:,c]
                    sums[c] += weight*(float(a[q])+grad(a,q,1,xx>0,xx<w-1)*dx+grad(a,q,w,yy>0,yy<h-1)*dy)
                total += weight
        if not total:
            raise RuntimeError('Fast marching stalled')
        out[i] = [byte(v/total) for v in sums]
        known[i] = True
        for j in neighbors(i,w,h):
            if not known[j]:
                offer(j)
    return image


def sources_and_patch(mask, patch):
    h, w = mask.shape
    if not isinstance(patch,int) or patch<3 or patch%2!=1 or patch>min(w,h):
        raise ValueError('Invalid patch')
    r = patch//2
    sources = [y*w+x for y in range(r,h-r) for x in range(r,w-r)
               if not mask[y-r:y+r+1,x-r:x+r+1].any()]
    if not sources and mask.any():
        raise ValueError('No complete source patch')
    return sources, r


def efros_leung(rgb, mask, patch=5, seed=1, tolerance=.1, **_):
    image, hole = validate(rgb, mask)
    if not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Invalid tolerance')
    sources, r = sources_and_patch(hole,patch)
    h,w = hole.shape
    src, out, hole = rgb.reshape(-1,3), image.reshape(-1,3), hole.ravel()
    draw = rng(seed)
    while hole.any():
        p,best,context = -1,-1,[]
        for i in range(w*h):
            if not hole[i] or not any(not hole[j] for j in neighbors(i,w,h)):
                continue
            x,y = i%w,i//w
            ctx,score = [],0.
            for dy in range(-r,r+1):
                for dx in range(-r,r+1):
                    xx,yy=x+dx,y+dy
                    if xx<0 or xx>=w or yy<0 or yy>=h or hole[yy*w+xx]:
                        continue
                    weight=math.exp(-(dx*dx+dy*dy)/(2*(patch/6.4)**2))
                    ctx.append((dy*w+dx,weight))
                    score+=weight
            if score>best:
                p,best,context=i,score,ctx
        distances=[]
        for q in sources:
            s=0.
            for offset,weight in context:
                for c in range(3):
                    s+=weight*(float(out[p+offset,c])-float(src[q+offset,c]))**2
            distances.append(s/(best*3))
        minimum=min(distances)
        eligible=[q for q,d in zip(sources,distances) if d<=minimum*(1+tolerance)+1e-12]
        out[p]=src[eligible[math.floor(draw()*len(eligible))]]
        hole[p]=False
    return image


def patchmatch(rgb, mask, patch=5, iterations=5, seed=1, **_):
    image,hole=validate(rgb,mask)
    sources,r=sources_and_patch(hole,patch)
    if not isinstance(iterations,int) or not 1<=iterations<=100:
        raise ValueError('Invalid iterations')
    if not hole.any():
        return image
    h,w=hole.shape
    src=rgb.reshape(-1,3)
    out=telea(rgb,mask).reshape(-1,3)
    targets=[y*w+x for y in range(r,h-r) for x in range(r,w-r)
             if hole[y-r:y+r+1,x-r:x+r+1].any()]
    if not targets:
        raise ValueError('No target patches')
    valid,active=set(sources),set(targets)
    draw=rng(seed)
    nnf={i:sources[math.floor(draw()*len(sources))] for i in targets}
    def distance(i,q):
        # Python integers avoid uint8 overflow.
        return sum((int(out[i+dy*w+dx,c])-int(src[q+dy*w+dx,c]))**2
                   for dy in range(-r,r+1) for dx in range(-r,r+1) for c in range(3))
    for it in range(iterations):
        costs={i:distance(i,nnf[i]) for i in targets}
        step=-1 if it%2 else 1
        for i in reversed(targets) if it%2 else targets:
            def offer(q):
                if q in valid:
                    d=distance(i,q)
                    if d<costs[i]:
                        costs[i],nnf[i]=d,q
            if 0<=i%w-step<w and i-step in active:
                q=nnf[i-step]
                if 0<=q%w+step<w:
                    offer(q+step)
            if i-step*w in active:
                offer(nnf[i-step*w]+step*w)
            ox,oy=nnf[i]%w,nnf[i]//w
            radius=max(w,h)
            while radius>=1:
                x=max(r,min(w-r-1,ox+math.floor(draw()*(2*radius+1))-radius))
                y=max(r,min(h-r-1,oy+math.floor(draw()*(2*radius+1))-radius))
                offer(y*w+x)
                radius//=2
        sums=np.zeros((w*h,3),dtype=np.float64)
        counts=np.zeros(w*h,dtype=np.int32)
        for i in targets:
            for dy in range(-r,r+1):
                for dx in range(-r,r+1):
                    p,q=i+dy*w+dx,nnf[i]+dy*w+dx
                    if hole.ravel()[p]:
                        sums[p]+=src[q]
                        counts[p]+=1
        for i in np.flatnonzero(counts):
            out[i]=[byte(v/counts[i]) for v in sums[i]]
    return out.reshape(h,w,3)


def inpaint(method, rgb, mask, **options):
    if method in ('criminisi','feature'):
        from exemplar import inpaint as exemplar_inpaint
        result,_=exemplar_inpaint(rgb,mask,method=method,**options)
        return result
    methods={'telea':telea,'bertalmio':bertalmio,'efros-leung':efros_leung,'patchmatch':patchmatch}
    if method not in methods:
        raise ValueError('Unknown method: '+method)
    return methods[method](rgb,mask,**options)


def bertalmio(rgb,mask,iterations=200,dt=.1,**_):
    image,hole=validate(rgb,mask)
    if not isinstance(iterations,int) or not 1<=iterations<=2000 or not math.isfinite(dt) or not 0<dt<=.2:
        raise ValueError('iterations 1..2000, dt (0,.2] required')
    a=telea(rgb,mask).astype(np.float64)/255
    def shifts(v):
        p=np.pad(v,((1,1),(1,1),(0,0)),mode='edge')
        return p[1:-1,:-2],p[1:-1,2:],p[:-2,1:-1],p[2:,1:-1]
    for _ in range(iterations):
        left,right,up,down=shifts(a)
        lap=left+right+up+down-4*a
        ll,lr,lu,ld=shifts(lap)
        ix,iy=(right-left)/2,(down-up)/2
        transport=(-iy*(lr-ll)/2+ix*(ld-lu)/2)/np.maximum(1e-3,np.hypot(ix,iy))
        update=np.clip(a+dt*(np.clip(transport,-.25,.25)+.2*lap),0,1)
        a[hole]=update[hole]
    image[hole]=np.floor(a[hole]*255+.5).clip(0,255).astype(np.uint8)
    return image
