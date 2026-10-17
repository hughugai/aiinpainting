/** Pure JS reference implementation. RGB is a flat 3-channel Uint8Array. */
export function inpaint(rgb, mask, width, height, options={}) {
  const {patch=9,lambda=.35,featureWeight=1,method='feature',gradientScale=255,
    confidenceUpdate='raw',maxIterations=mask.length+1,onProgress=()=>{}}=options;
  const w=width,h=height,n=w*h,r=Math.floor(patch/2);
  if (!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||rgb.length!==n*3||mask.length!==n)
    throw Error('Invalid dimensions or array lengths');
  if (!Number.isInteger(patch)||patch<3||patch%2!==1||patch>Math.min(w,h)) throw Error('Invalid patch size');
  if (![lambda,featureWeight,gradientScale].every(Number.isFinite)||gradientScale<=0||!Number.isInteger(maxIterations)||maxIterations<0)
    throw Error('Invalid numeric options');
  if (!['feature','shifted','criminisi'].includes(method)||!['raw','refined'].includes(confidenceUpdate)) throw Error('Invalid mode');
  const image=Uint8Array.from(rgb),hole=Uint8Array.from(mask,v=>v?1:0),initial=hole.slice();
  const confidence=Float64Array.from(hole,v=>1-v),trace=[],cache=new Map();
  let remaining=hole.reduce((a,b)=>a+b,0);
  const neighbors=(y,x)=>[[y-1,x],[y+1,x],[y,x-1],[y,x+1]].filter(([yy,xx])=>yy>=0&&yy<h&&xx>=0&&xx<w);
  function sources(ph,pw) {
    const key=ph+','+pw;
    if (!cache.has(key)) {
      const list=[];
      for(let y=0;y<=h-ph;y++) for(let x=0;x<=w-pw;x++) {
        let valid=true;
        for(let dy=0;dy<ph&&valid;dy++) for(let dx=0;dx<pw;dx++) if(initial[(y+dy)*w+x+dx]) {valid=false;break;}
        if(valid) list.push([y,x]);
      }
      cache.set(key,list);
    }
    return cache.get(key);
  }
  for(let iteration=0;iteration<maxIterations&&remaining;iteration++) {
    const gray=new Float64Array(n),gx=new Float64Array(n),gy=new Float64Array(n),mag=new Float64Array(n);
    for(let i=0;i<n;i++) gray[i]=image[3*i]*.299+image[3*i+1]*.587+image[3*i+2]*.114;
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
      const i=y*w+x;if(hole[i]) continue;
      const l=x>0&&!hole[i-1],rr=x+1<w&&!hole[i+1],u=y>0&&!hole[i-w],d=y+1<h&&!hole[i+w];
      gx[i]=l&&rr?(gray[i+1]-gray[i-1])/2:rr?gray[i+1]-gray[i]:l?gray[i]-gray[i-1]:0;
      gy[i]=u&&d?(gray[i+w]-gray[i-w])/2:d?gray[i+w]-gray[i]:u?gray[i]-gray[i-w]:0;
      mag[i]=Math.hypot(gx[i],gy[i]);
    }
    let best=null;
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
      const i=y*w+x,nb=neighbors(y,x);
      if(hole[i]||!nb.some(([yy,xx])=>hole[yy*w+xx])) continue;
      const y0=Math.max(0,y-r),y1=Math.min(h,y+r+1),x0=Math.max(0,x-r),x1=Math.min(w,x+r+1);
      let c=0,maxMag=-1,maxI=0;
      for(let yy=y0;yy<y1;yy++) for(let xx=x0;xx<x1;xx++) {
        const j=yy*w+xx;c+=confidence[j];if(mag[j]>maxMag) {maxMag=mag[j];maxI=j;}
      }
      c/=(y1-y0)*(x1-x0);
      let nx=((x+1<w?hole[i+1]:0)-(x>0?hole[i-1]:0))/2;
      let ny=((y+1<h?hole[i+w]:0)-(y>0?hole[i-w]:0))/2;
      let norm=Math.hypot(nx,ny);
      if(norm===0) {const [yy,xx]=nb.find(([yy,xx])=>hole[yy*w+xx]);nx=xx-x;ny=yy-y;norm=Math.hypot(nx,ny);}
      nx/=norm;ny/=norm;
      const data=Math.abs(-gy[maxI]*nx+gx[maxI]*ny)/255,feature=mag[i]/gradientScale;
      let refined=method==='criminisi'?c:c-lambda;if(method==='feature') refined+=featureWeight*feature;
      const priority=refined*data;
      if(!best||priority>best.priority) best={priority,y,x,y0,y1,x0,x1,c,data,feature,refined};
    }
    if(!best) throw Error('No known fill front (mask may cover entire image)');
    const {priority,y,x,y0,y1,x0,x1,c,data,feature,refined}=best,ph=y1-y0,pw=x1-x0;
    const known=[];for(let dy=0;dy<ph;dy++) for(let dx=0;dx<pw;dx++) if(!hole[(y0+dy)*w+x0+dx]) known.push([dy,dx]);
    let minSSD=Infinity,source=null;
    for(const [sy,sx] of sources(ph,pw)) {
      let ssd=0;
      for(const [dy,dx] of known) {
        const ti=((y0+dy)*w+x0+dx)*3,si=((sy+dy)*w+sx+dx)*3;
        for(let ch=0;ch<3;ch++) {const delta=image[ti+ch]-image[si+ch];ssd+=delta*delta;}
        if(ssd>minSSD) break;
      }
      if(ssd<minSSD) {minSSD=ssd;source=[sy,sx];}
    }
    if(!source) throw Error('No fully known source patch; reduce patch size');
    const [sy,sx]=source;let filled=0;
    for(let dy=0;dy<ph;dy++) for(let dx=0;dx<pw;dx++) {
      const ti=(y0+dy)*w+x0+dx;if(!hole[ti]) continue;
      const si=(sy+dy)*w+sx+dx;for(let ch=0;ch<3;ch++) image[3*ti+ch]=image[3*si+ch];
      confidence[ti]=confidenceUpdate==='raw'?c:refined;hole[ti]=0;filled++;
    }
    remaining-=filled;
    const step={iteration,center:[x,y],source:[sx,sy],confidence:c,data,feature,refined,priority,ssd:minSSD,filled};
    trace.push(step);onProgress({remaining,step});
  }
  if(remaining) throw Error('Iteration limit reached before completion');
  return {image,trace};
}
