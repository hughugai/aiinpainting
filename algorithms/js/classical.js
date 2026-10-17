/** Browser/Node reference algorithms. RGB bytes, mask!=0 means hole. */
import { inpaint as exemplar } from './exemplar.js';
export function validate(rgb, mask, w, h) {
  if (!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||rgb.length!==w*h*3||mask.length!==w*h) throw Error('Invalid RGB/mask dimensions');
  if ([...mask].every(Boolean)) throw Error('At least one known pixel is required');
}
const byte=v=>Math.max(0,Math.min(255,Math.floor(v+.5)));
function rng(seed=1) { let s=seed>>>0; return ()=>((s=(Math.imul(1664525,s)+1013904223)>>>0)/4294967296); }
const adjacent=(i,w,h)=>[i%w?i-1:-1,i%w<w-1?i+1:-1,i>=w?i-w:-1,i<w*(h-1)?i+w:-1].filter(j=>j>=0);

/** Telea-style fast marching: Eikonal arrival times + first-order image extrapolation.
 * Discretization differs from OpenCV: known exterior T=0, accepted-only colors.
 */
export function telea(rgb, mask, w, h, {radius=3}={}) {
  validate(rgb,mask,w,h);
  if (!Number.isInteger(radius)||radius<1||radius>20) throw Error('radius must be 1..20');
  const n=w*h,out=Uint8Array.from(rgb),known=Uint8Array.from(mask,v=>v?0:1);
  const t=Float64Array.from(mask,v=>v?Infinity:0),heap=[];
  let serial=0;
  const before=(a,b)=>a[0]<b[0]||(a[0]===b[0]&&a[2]<b[2]);
  function push(v) { heap.push(v);let i=heap.length-1;while(i) {const p=(i-1)>>1;if(!before(v,heap[p]))break;heap[i]=heap[p];i=p;}heap[i]=v; }
  function pop() {const first=heap[0],last=heap.pop();if(heap.length) {let i=0;while(2*i+1<heap.length){let j=2*i+1;if(j+1<heap.length&&before(heap[j+1],heap[j]))j++;if(!before(heap[j],last))break;heap[i]=heap[j];i=j;}heap[i]=last;}return first;}
  function arrival(i) {
    const x=i%w,y=Math.floor(i/w);
    const a=Math.min(x&&known[i-1]?t[i-1]:Infinity,x<w-1&&known[i+1]?t[i+1]:Infinity);
    const b=Math.min(y&&known[i-w]?t[i-w]:Infinity,y<h-1&&known[i+w]?t[i+w]:Infinity);
    if(!Number.isFinite(a))return b+1;if(!Number.isFinite(b))return a+1;
    return Math.abs(a-b)>=1?Math.min(a,b)+1:(a+b+Math.sqrt(2-(a-b)**2))/2;
  }
  function offer(i) {const v=arrival(i);if(v<t[i]){t[i]=v;push([v,i,serial++]);}}
  for(let i=0;i<n;i++)if(!known[i]&&adjacent(i,w,h).some(j=>known[j]))offer(i);
  function grad(a,i,step,low,high) {
    const l=low&&known[i-step],r=high&&known[i+step];
    return l&&r?(a(i+step)-a(i-step))/2:r?a(i+step)-a(i):l?a(i)-a(i-step):0;
  }
  while(heap.length) {
    const [d,i]=pop();if(known[i]||d!==t[i])continue;
    const x=i%w,y=Math.floor(i/w),tv=j=>t[j];
    const nx=grad(tv,i,1,x>0,x<w-1),ny=grad(tv,i,w,y>0,y<h-1);
    const sums=[0,0,0];let total=0;
    for(let yy=Math.max(0,y-radius);yy<=Math.min(h-1,y+radius);yy++)for(let xx=Math.max(0,x-radius);xx<=Math.min(w-1,x+radius);xx++) {
      const q=yy*w+xx,dx=x-xx,dy=y-yy,dist=Math.hypot(dx,dy);
      if(!known[q]||dist===0||dist>radius)continue;
      const weight=(Math.abs(dx*nx+dy*ny)+1e-6)/(dist**3*(1+Math.abs(d-t[q])));
      for(let c=0;c<3;c++){const a=j=>out[3*j+c];sums[c]+=weight*(a(q)+grad(a,q,1,xx>0,xx<w-1)*dx+grad(a,q,w,yy>0,yy<h-1)*dy);}
      total+=weight;
    }
    if(!total)throw Error('Fast marching stalled');
    for(let c=0;c<3;c++)out[3*i+c]=byte(sums[c]/total);
    known[i]=1;for(const j of adjacent(i,w,h))if(!known[j])offer(j);
  }
  return out;
}

/** Gaussian neighborhood conditional sampling (Efros-Leung, 1999).
 * Hole-completion adaptation: fixed source, strongest-context frontier, seeded draws.
 */
export function efrosLeung(rgb, mask,w,h,{patch=5,seed=1,tolerance=.1}={}) {
  validate(rgb,mask,w,h);
  if(!Number.isInteger(patch)||patch<3||patch%2!==1||patch>Math.min(w,h)||!Number.isFinite(tolerance)||tolerance<0)throw Error('Invalid patch/tolerance');
  const out=Uint8Array.from(rgb),hole=Uint8Array.from(mask,Boolean),r=patch>>1,random=rng(seed),sources=[];
  for(let y=r;y<h-r;y++)for(let x=r;x<w-r;x++) {let ok=true;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++)if(mask[(y+dy)*w+x+dx])ok=false;if(ok)sources.push(y*w+x);}
  if(!sources.length&&hole.some(Boolean))throw Error('No complete source patch');
  let remaining=hole.reduce((a,b)=>a+b,0);
  while(remaining) {
    let p=-1,best=-1,context=[];
    for(let i=0;i<w*h;i++)if(hole[i]&&adjacent(i,w,h).some(j=>!hole[j])) {
      const x=i%w,y=Math.floor(i/w),ctx=[];let score=0;
      for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++) {const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h||hole[yy*w+xx])continue;const weight=Math.exp(-(dx*dx+dy*dy)/(2*(patch/6.4)**2));ctx.push([dy*w+dx,weight]);score+=weight;}
      if(score>best){best=score;p=i;context=ctx;}
    }
    let min=Infinity;const distances=sources.map(q=>{let s=0;for(const [offset,weight]of context)for(let c=0;c<3;c++)s+=weight*(out[3*(p+offset)+c]-rgb[3*(q+offset)+c])**2;s/=best*3;min=Math.min(min,s);return s;});
    const eligible=sources.filter((_,j)=>distances[j]<=min*(1+tolerance)+1e-12),q=eligible[Math.floor(random()*eligible.length)];
    out.set(rgb.slice(q*3,q*3+3),p*3);hole[p]=0;remaining--;
  }
  return out;
}

/** PatchMatch NNF with alternating propagation and shrinking random search.
 * Completion uses iterative patch voting, an engineering reconstruction wrapper,
 * not the full multiscale interactive editing system in Barnes et al. (2009).
 */
export function patchMatch(rgb,mask,w,h,{patch=5,iterations=5,seed=1}={}) {
  validate(rgb,mask,w,h);
  if(!Number.isInteger(patch)||patch<3||patch%2!==1||patch>Math.min(w,h)||!Number.isInteger(iterations)||iterations<1||iterations>100)throw Error('Invalid patch/iterations');
  const r=patch>>1,n=w*h,random=rng(seed),out=telea(rgb,mask,w,h),sources=[],valid=new Uint8Array(n),nnf=new Int32Array(n),cost=new Float64Array(n);
  const targets=[];
  for(let y=r;y<h-r;y++)for(let x=r;x<w-r;x++) {const i=y*w+x;let clean=true,touches=false;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){if(mask[i+dy*w+dx]){clean=false;touches=true;}}if(clean){valid[i]=1;sources.push(i);}if(touches)targets.push(i);}
  if(!mask.some(Boolean))return out;
  if(!sources.length||!targets.length)throw Error('No complete source/target patches');
  function distance(i,q){let sum=0;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++)for(let c=0;c<3;c++)sum+=(out[3*(i+dy*w+dx)+c]-rgb[3*(q+dy*w+dx)+c])**2;return sum;}
  for(const i of targets)nnf[i]=sources[Math.floor(random()*sources.length)];
  for(let it=0;it<iterations;it++) {
    for(const i of targets)cost[i]=distance(i,nnf[i]);
    const order=it%2?[...targets].reverse():targets,step=it%2?-1:1;
    const active=new Set(targets);
    for(const i of order) {
      const offer=q=>{if(q<0||q>=n||!valid[q])return;const d=distance(i,q);if(d<cost[i]){cost[i]=d;nnf[i]=q;}};
      if(i%w-step>=0&&i%w-step<w&&active.has(i-step)) {const q=nnf[i-step];if(q%w+step>=0&&q%w+step<w)offer(q+step);}
      if(active.has(i-step*w))offer(nnf[i-step*w]+step*w);
      const origin=nnf[i],ox=origin%w,oy=Math.floor(origin/w);
      for(let radius=Math.max(w,h);radius>=1;radius=Math.floor(radius/2)) {
        const x=Math.max(r,Math.min(w-r-1,ox+Math.floor(random()*(2*radius+1))-radius));
        const y=Math.max(r,Math.min(h-r-1,oy+Math.floor(random()*(2*radius+1))-radius));offer(y*w+x);
      }
    }
    const sums=new Float64Array(n*3),counts=new Uint32Array(n);
    for(const i of targets)for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const p=i+dy*w+dx,q=nnf[i]+dy*w+dx;if(mask[p]){counts[p]++;for(let c=0;c<3;c++)sums[p*3+c]+=rgb[q*3+c];}}
    for(let i=0;i<n;i++)if(counts[i])for(let c=0;c<3;c++)out[i*3+c]=byte(sums[i*3+c]/counts[i]);
  }
  return out;
}
export function inpaint(method,rgb,mask,w,h,options={}) {
  if(method==='criminisi'||method==='feature')return exemplar(rgb,mask,w,h,{...options,method}).image;
  const f={telea,bertalmio,'efros-leung':efrosLeung,patchmatch:patchMatch}[method];
  if(!f)throw Error('Unknown method: '+method);
  return f(rgb,mask,w,h,options);
}

/** Bertalmio-inspired isophote transport of the Laplacian.
 * Educational discretization, NOT the full SIGGRAPH 2000 anisotropic-diffusion solver.
 * Normalized transport with a clipped explicit step and harmonic regularization.
 */
export function bertalmio(rgb,mask,w,h,{iterations=200,dt=.1}={}) {
  validate(rgb,mask,w,h);
  if(!Number.isInteger(iterations)||iterations<1||iterations>2000||!Number.isFinite(dt)||dt<=0||dt>.2)throw Error('iterations 1..2000, dt (0,.2] required');
  let a=Float64Array.from(telea(rgb,mask,w,h),v=>v/255);
  const at=(b,x,y,c)=>b[3*(Math.max(0,Math.min(h-1,y))*w+Math.max(0,Math.min(w-1,x)))+c];
  for(let it=0;it<iterations;it++) {
    const lap=new Float64Array(a.length),next=a.slice();
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)for(let c=0;c<3;c++)lap[3*(y*w+x)+c]=at(a,x-1,y,c)+at(a,x+1,y,c)+at(a,x,y-1,c)+at(a,x,y+1,c)-4*at(a,x,y,c);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x])for(let c=0;c<3;c++) {
      const i=3*(y*w+x)+c,ix=(at(a,x+1,y,c)-at(a,x-1,y,c))/2,iy=(at(a,x,y+1,c)-at(a,x,y-1,c))/2;
      const lx=(at(lap,x+1,y,c)-at(lap,x-1,y,c))/2,ly=(at(lap,x,y+1,c)-at(lap,x,y-1,c))/2;
      const transport=(-iy*lx+ix*ly)/Math.max(1e-3,Math.hypot(ix,iy));
      next[i]=Math.max(0,Math.min(1,a[i]+dt*(Math.max(-.25,Math.min(.25,transport))+.2*lap[i])));
    }
    a=next;
  }
  const out=Uint8Array.from(rgb);for(let i=0;i<w*h;i++)if(mask[i])for(let c=0;c<3;c++)out[3*i+c]=byte(a[3*i+c]*255);return out;
}
