import {inpaint} from '../inpaint.js';
self.onmessage=({data:a})=>{try {
 const result=inpaint(a.rgb,a.mask,a.width,a.height,{...a.options,onProgress:p=>self.postMessage({progress:p.remaining})});
 self.postMessage({result});
}catch(e){self.postMessage({error:e.message});}};
